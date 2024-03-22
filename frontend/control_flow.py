import dataclasses
from typing import Any, Optional, TYPE_CHECKING, Callable
import enum
import torch
import operator
import typing
from .store_pos import StorePos
from .c_api import parse_cell, set_cell
from .variables import Variable, TensorVar
from .pycode_writer import new_name
if TYPE_CHECKING:
    from .guard_tracker import State


@dataclasses.dataclass
class LoopPosMap:
    input_only_pos: list[tuple[str, StorePos]]
    joint_pos: list[tuple[str, StorePos]]
    output_only_pos: list[tuple[str, StorePos]]


class LoopModule(torch.nn.Module):  #type: ignore
    body: torch.fx.GraphModule
    num_read_only_param: int
    num_iter: int

    def __init__(self, body: torch.fx.GraphModule, num_read_only_param: int,
                 num_iter: int):
        super(LoopModule, self).__init__()
        self.body = body
        self.num_read_only_param = num_read_only_param
        self.num_iter = num_iter

    # def forward(self, num_iter: Optional[int], cond: torch.Tensor, *values:
    #             Any) -> Any:
    def forward(self, *values: Any) -> Any:
        iter_num = 0
        # assert cond.dtype == torch.bool
        read_only = values[:self.num_read_only_param]
        loop_carry = values[self.num_read_only_param:]
        while iter_num < self.num_iter:
            # and cond.item():
            loop_carry = self.body(torch.tensor(iter_num), *read_only,
                                   *loop_carry)
            # cond, *loop_carry = self.body(iter_num, cond, *read_only,
            #                               *loop_carry)
            iter_num += 1
        return loop_carry


class CondModule(torch.nn.Module):  # type: ignore
    true_body: torch.fx.GraphModule
    false_body: torch.fx.GraphModule

    def __init__(self, true_body: torch.fx.GraphModule,
                 false_body: torch.fx.GraphModule) -> None:
        super().__init__()
        self.true_body = true_body
        self.false_body = false_body

    def forward(self, cond: torch.Tensor, *values: Any) -> Any:
        if cond.item():
            return self.true_body(cond, *values)
        else:
            return self.false_body(cond, *values)


class ControlFlowInfo:
    start_pc: int
    end_pc: int

    def __init__(self, start_pc: int, end_pc: int) -> None:
        self.start_pc = start_pc
        self.end_pc = end_pc


class ForLoopInfo(ControlFlowInfo):
    num_iter: int
    cur_iter: int
    pos_map: Optional[LoopPosMap]
    inner_graph: Optional[torch.fx.Graph]

    def __init__(self, start_pc: int, end_pc: int, num_iter: int) -> None:
        super().__init__(start_pc, end_pc)
        self.num_iter = num_iter
        self.cur_iter = 0


@dataclasses.dataclass
class RecoverInfo:
    node_mapping: dict[torch.fx.Node, torch.fx.Node]
    objects: dict[str, object]
    subgraph: torch.fx.Graph


class IfStmtInfo(ControlFlowInfo):
    if_true_fn: Callable[..., Any]
    if_false_fn: Callable[..., Any]
    cond_obj: Any
    cond: bool
    cells: dict[str, Any]
    cell_values: dict[str, Any]
    recover_info: Optional[RecoverInfo]
    frame_root: torch.nn.Module

    class State(enum.Enum):
        NOT_CALLING = 0
        CALLING_OTHER_BRANCH = 1
        IN_THE_MIDDLE = 2
        CALLING_RUN_BRANCH = 3

    state: Optional["State"]
    stored_locals: Optional[set[str]]

    def __init__(self, start_pc: int, end_pc: int, if_true_fn: Callable[...,
                                                                        Any],
                 if_false_fn: Callable[..., Any], cond_obj: Any, cond: bool,
                 frame_root: torch.nn.Module) -> None:
        super().__init__(start_pc, end_pc)
        self.if_true_fn = if_true_fn
        self.if_false_fn = if_false_fn
        self.cond_obj = cond_obj
        self.cond = cond
        self.state = self.State.NOT_CALLING
        self.recover_info = None
        self.frame_root = frame_root
        self.parse_cells()

    def parse_cells(self) -> None:
        if_true = self.if_true_fn
        if_false = self.if_false_fn
        assert if_true.__closure__ is not None
        assert if_false.__closure__ is not None
        cells = {
            if_true.__code__.co_freevars[i]: if_true.__closure__[i]
            for i in range(len(if_true.__code__.co_freevars))
        }
        for i in range(len(if_false.__code__.co_freevars)):
            var_name = if_false.__code__.co_freevars[i]
            cell = if_false.__closure__[i]
            if var_name in cells:
                assert id(cells[var_name]) == id(cell)
            else:
                cells[var_name] = cell
        self.cells = cells
        self.cell_values = {
            name: parse_cell(cell) for name, cell in cells.items()
        }

    def mark_end(self, stored_locals: set[str], full_fx_graph: torch.fx.Graph,
                 get_var: Callable[..., Variable]) -> None:
        self.stored_locals = stored_locals
        if self.state == self.State.CALLING_OTHER_BRANCH:
            self.state = self.State.IN_THE_MIDDLE
            graph = torch.fx.Graph()
            node_mapping: dict[torch.fx.Node, torch.fx.Node] = {}
            for node in full_fx_graph.nodes:
                new_node = graph.node_copy(node,
                                           lambda node: node_mapping[node])
                node_mapping[node] = new_node
            for node in reversed(full_fx_graph.nodes):
                if node.op == "placeholder":
                    continue
                full_fx_graph.erase_node(node)
            assert self.recover_info is None
            self.recover_info = RecoverInfo(node_mapping, {}, graph)
        elif self.state == self.State.CALLING_RUN_BRANCH:
            self.state = self.State.NOT_CALLING
            run1_info = self.recover_info
            run2_info = RecoverInfo({}, {}, torch.fx.Graph())
            assert run1_info is not None
            for node in full_fx_graph.nodes:
                new_node = run2_info.subgraph.node_copy(
                    node, lambda node: run2_info.node_mapping[node])
                run2_info.node_mapping[node] = new_node
            for node in full_fx_graph.nodes:
                if node.op == "placeholder" and node not in run1_info.node_mapping:
                    new_node = run1_info.subgraph.node_copy(
                        node, lambda node: run1_info.node_mapping[node])
                    run1_info.node_mapping[node] = new_node
            # objects_new: dict[str, Any] = {}
            for name in stored_locals:
                if name in self.cell_values:
                    run2_info.objects[name] = parse_cell(self.cells[name])
            all_stored_cells = list(
                set(run1_info.objects.keys()) | set(run2_info.objects.keys()))
            run1_output: list[torch.fx.Node] = []
            run2_output: list[torch.fx.Node] = []
            tensor_vars: list[TensorVar] = []
            for k in all_stored_cells:
                if k not in run1_info.objects:
                    run1_info.objects[k] = self.cell_values[k]
                if k not in run2_info.objects:
                    run2_info.objects[k] = self.cell_values[k]
                if isinstance(run1_info.objects[k], torch.Tensor):
                    assert isinstance(run2_info.objects[k], torch.Tensor)
                    run1_var = get_var(run1_info.objects[k])
                    assert isinstance(run1_var, TensorVar)
                    run1_node = run1_var.as_fx_node()
                    assert isinstance(run1_node, torch.fx.Node)
                    run1_output.append(run1_info.node_mapping[run1_node])

                    run2_var = get_var(run2_info.objects[k])
                    assert isinstance(run2_var, TensorVar)
                    run2_node = run2_var.as_fx_node()
                    assert isinstance(run2_node, torch.fx.Node)
                    run2_output.append(run2_info.node_mapping[run2_node])
                    tensor_vars.append(run2_var)
                elif run1_info.objects[k] == run2_info.objects[k]:
                    continue
                else:
                    print("run info", k, run1_info.objects[k],
                          run2_info.objects[k])
                    raise NotImplementedError
            run1_info.subgraph.output(tuple(run1_output))
            run2_info.subgraph.output(tuple(run2_output))
            run1_module = torch.fx.GraphModule(self.frame_root,
                                               run1_info.subgraph)
            run2_module = torch.fx.GraphModule(self.frame_root,
                                               run2_info.subgraph)
            if self.cond:
                if_true_module = run2_module
                if_false_module = run1_module
            else:
                if_true_module = run1_module
                if_false_module = run2_module
            cond_module = CondModule(if_true_module, if_false_module)
            cond_module_name = new_name("__cond_module__")
            self.frame_root.add_module(cond_module_name, cond_module)
            # TODO: handle in outside
            # self.state.submodule_paths[cond_module] = cond_module_name
            cond_inputs = [
                x for x in full_fx_graph.nodes if x.op == "placeholder"
            ]
            for node in reversed(full_fx_graph.nodes):
                if node.op != "placeholder":
                    full_fx_graph.erase_node(node)
            cond_node = full_fx_graph.call_module(cond_module_name,
                                                  tuple(cond_inputs))
            for i, tensor_var in enumerate(tensor_vars):
                new_node = full_fx_graph.call_function(operator.getitem,
                                                       (cond_node, i))
                new_node.meta["var"] = tensor_var
                tensor_var.fx_node = new_node
        else:
            raise ValueError(f"state {self.state} not expected")

    def mark_start(self) -> None:
        if self.state == self.State.NOT_CALLING:
            self.state = self.State.CALLING_OTHER_BRANCH
        elif self.state == self.State.IN_THE_MIDDLE:
            self.state = self.State.CALLING_RUN_BRANCH
        else:
            raise ValueError(f"state {self.state} not expected")

    def recover(self) -> None:
        assert self.state == self.State.IN_THE_MIDDLE
        assert self.stored_locals is not None
        assert self.recover_info is not None
        for name in self.stored_locals:
            if name in self.cell_values:
                self.recover_info.objects[name] = parse_cell(self.cells[name])
                set_cell(self.cells[name], self.cell_values[name])


class TraceError(Exception):
    pass


def recover() -> None:  # an empty function, handled in tracer
    pass


def merge() -> None:
    pass


def break_at_callsite() -> None:  # NOTE: unimplemented
    pass


def fake_branch_call0(*_args: Any, **_kwargs: Any) -> None:
    pass


def fake_branch_call1(*_args: Any, **_kwargs: Any) -> None:
    pass


# mypy: disable-error-code="name-defined"
def if_stmt(cond: bool, if_true: Callable[..., Any],
            if_false: Callable[..., Any]) -> Any:
    try:
        if_other_branch()
    except Exception as e:
        break_at_callsite()
    recover()
    return if_run_branch()
