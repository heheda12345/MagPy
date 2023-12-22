import dataclasses
from typing import Any, Optional, TYPE_CHECKING, Callable
import enum
import torch
from .store_pos import StorePos
from .c_api import parse_cell, set_cell
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
            return self.true_body(*values)
        else:
            return self.false_body(*values)


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


class IfStmtInfo(ControlFlowInfo):
    if_true_fn: Callable[..., Any]
    if_false_fn: Callable[..., Any]
    cond: bool
    cells: dict[str, Any]
    cell_values: dict[str, Any]

    class State(enum.Enum):
        NOT_CALLING = 0
        CALLING_OTHER_BRANCH = 1
        IN_THE_MIDDLE = 2
        CALLING_RUN_BRANCH = 3

    state: Optional["State"]
    stored_locals: Optional[set[str]]

    def __init__(self, start_pc: int, end_pc: int, if_true_fn: Callable[...,
                                                                        Any],
                 if_false_fn: Callable[..., Any], cond: bool) -> None:
        super().__init__(start_pc, end_pc)
        self.if_true_fn = if_true_fn
        self.if_false_fn = if_false_fn
        self.cond = cond
        self.state = self.State.NOT_CALLING
        self.parse_cells()

    def parse_cells(self) -> None:
        if_true = self.if_true_fn
        if_false = self.if_false_fn
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

    def mark_end(self, stored_locals: set[str]) -> None:
        self.stored_locals = stored_locals
        if self.state == self.State.CALLING_OTHER_BRANCH:
            self.state = self.State.IN_THE_MIDDLE
        elif self.state == self.State.CALLING_RUN_BRANCH:
            self.state = self.State.NOT_CALLING
        else:
            raise ValueError(f"state {self.state} not expected")

    def mark_start(self) -> None:
        if self.state == self.State.NOT_CALLING:
            self.state = self.State.CALLING_OTHER_BRANCH
        elif self.state == self.State.IN_THE_MIDDLE:
            self.state = self.State.CALLING_RUN_BRANCH
        else:
            raise ValueError(f"state {self.state} not expected")

    def recover(self):
        assert self.state == self.State.IN_THE_MIDDLE
        assert self.stored_locals is not None
        for name in self.stored_locals:
            if name in self.cell_values:
                set_cell(self.cells[name], self.cell_values[name])


class TraceError(Exception):
    pass


def recover() -> None:  # an empty function, handled in tracer
    pass


def break_at_callsite() -> None:  # NOTE: unimplemented
    pass


def fake_branch_call0(*_args: Any, **_kwargs: Any) -> None:
    pass


def fake_branch_call1(*_args: Any, **_kwargs: Any) -> None:
    pass


def if_stmt(cond, if_true, if_false):
    try:
        if_other_branch()
    except Exception as e:
        break_at_callsite()
    recover()  # TODO: implement recover
    return if_run_branch()
