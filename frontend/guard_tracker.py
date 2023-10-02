from types import FrameType
from typing import Dict, Any, Callable, List, Optional
import inspect
import logging
import itertools
import torch
import torch.fx
import operator
import dis
import traceback
import copy
from .code import ProcessedCode, load_code
from .c_api import get_value_stack_from_top, get_value_stack_size, set_eval_frame, stack_effect
from .instruction import Instruction, ci
from .cache import CachedGraph, get_frame_cache
from .store_pos import StorePos, StoreInStack, StoreInLocal, StoreInGlobal, StoreInAttr, StoreInIndex
from . import variables as vs
from .utils import is_scalar, new_random_key, has_force_graph_break, NullObject, is_call_bytecode, fx_graph_functions, fx_graph_inplace_functions, is_user_defined_func, UnknownTypeError, get_all_objects_in_stack
from .object_table import ObjectTable
from .pycode_generator import GraphFnCodegen, GuardFnCodegen
from .fx_graph import FxGraph, get_frame_root, is_leaf_module, NodeArgs
from .bytecode_analysis import livevars_analysis
from .variables.tuple_ import TupleVar
from .variables.base import Variable


class State:
    objects: ObjectTable
    start_pc: int
    start_stack_size: int
    is_empty: bool
    fx_graph: FxGraph
    root: torch.nn.Module
    node_waiting_ids: list[torch.fx.Node]
    stored_locals: set[str]
    stored_globals: set[str]
    submodule_paths: dict[torch.nn.Module, str]
    subparam_paths: dict[torch.nn.Parameter, str]
    written: bool
    defer_restart: Optional[str]  # None if no need to restart
    stack_objs: Optional[list[Any]]
    object_refs: list[Any]  # hold the reference of objects to avoid GC
    inplace_update_objs: list[Any]

    def __init__(self, root: torch.nn.Module) -> None:
        self.objects = ObjectTable()
        self.start_pc = -1
        self.start_stack_size = -1
        self.is_empty = True

        def get_mark_written_fn(state: 'State') -> Callable[[], None]:
            return lambda: setattr(state, "written", True)

        self.fx_graph = FxGraph(root, get_mark_written_fn(self))
        self.root = root
        self.node_waiting_ids = []
        self.stored_locals = set()
        self.stored_globals = set()
        self.submodule_paths = {}
        self.subparam_paths = {}
        self.update_subpath(root, "")

        self.written = False
        self.defer_restart = None
        self.stack_objs = None
        self.object_refs = []
        self.num_new_refs = 0
        self.inplace_update_objs = []

    def update_subpath(self, module: torch.nn.Module, prefix: str) -> None:

        def get_name(prefix: str, name: str) -> str:
            if prefix == "":
                return name
            if name == "":
                return prefix
            return prefix + "." + name

        for name, mod in module.named_modules():
            self.submodule_paths[mod] = get_name(prefix, name)
        for name, param in module.named_parameters():
            self.subparam_paths[param] = get_name(prefix, name)

    def add_submodule(self, module: torch.nn.Module) -> None:
        new_module_name = "__external__" + str(len(self.submodule_paths))
        self.update_subpath(module, new_module_name)
        self.root.add_module(new_module_name, module)
        self.update_subpath(module, new_module_name)
        # self.written = True # not mark as written as graph break may happen

    def as_node_args_kwargs(
        self, args: list[Any], kwargs: dict[str, Any]
    ) -> tuple[tuple[torch.fx.Node, ...], dict[str, torch.fx.Node]]:

        def as_fx_node(var: vs.Variable) -> NodeArgs:
            if isinstance(var, vs.TorchParamVar):
                return self.fx_graph.create_node("get_attr",
                                                 self.subparam_paths[var.obj],
                                                 (), {})
            return var.as_fx_node()

        node_args = tuple(
            as_fx_node(self.objects.get(arg, allow_unexist_const=True))
            for arg in args)
        node_kwargs = {
            key: as_fx_node(self.objects.get(arg, allow_unexist_const=True))
            for key, arg in kwargs.items()
        }

        return node_args, node_kwargs

    def record_function(self,
                        func: Callable[..., Any],
                        args: List[Any],
                        kwargs: Dict[str, Any],
                        add_proxy_waiting_id: bool = True) -> None:
        pargs, pkwargs = self.as_node_args_kwargs(args, kwargs)
        self.written = True
        if isinstance(func, torch.nn.Module):
            if is_leaf_module(func):
                fx_node = self.fx_graph.create_node(
                    "call_module",
                    self.submodule_paths[func],
                    pargs,
                    pkwargs,
                )
                if add_proxy_waiting_id:
                    self.node_waiting_ids.append(fx_node)
            else:
                for k, v in self.submodule_paths.items():
                    print(id(k), v, k)
                raise NotImplementedError(func)
        else:
            fx_node = self.fx_graph.create_node(
                "call_function",
                func,
                pargs,
                pkwargs,
            )
            if add_proxy_waiting_id:
                self.node_waiting_ids.append(fx_node)

    @classmethod
    def from_frame(cls, frame: FrameType, read_stack: bool,
                   frame_root: torch.nn.Module) -> 'State':
        state = cls(frame_root)
        if read_stack:
            state.start_stack_size = get_value_stack_size(frame)
            for i in range(state.start_stack_size):
                value = get_value_stack_from_top(frame, i)
                var = vs.make_var_from_value(value, True,
                                             state.objects.get_or_make_var,
                                             state.fx_graph,
                                             [StoreInLocal(f"__stack__{i}")])
                state.objects.add(var, value)
        # state.written may be assigned inside make_var_from_value
        state.written = False
        return state

    def add_object(self, var: vs.Variable, value: Any) -> None:
        self.written = True
        self.objects.add(var, value)

    def add_stored_locals(self, name: str) -> None:
        self.written = True
        self.stored_locals.add(name)

    def add_stored_globals(self, name: str) -> None:
        self.written = True
        self.stored_globals.add(name)

    def store_pos_in_callee(self, pos: StorePos, idx: int) -> StorePos:
        if idx in self.objects.objs:
            var = self.objects.objs[idx]
            return var.extract_code_at_start[0]
        if isinstance(pos, StoreInLocal):
            raise ValueError("unknown local in callee")
        elif isinstance(pos, StoreInStack):
            raise ValueError("cannot store in stack in callee")
        elif isinstance(pos, StoreInGlobal):
            return pos
        elif isinstance(pos, StoreInAttr):
            return StoreInAttr(
                self.store_pos_in_callee(pos.self_pos, pos.self_id),
                pos.self_id, pos.attr_name)
        elif isinstance(pos, StoreInIndex):
            return StoreInIndex(
                self.store_pos_in_callee(pos.self_pos, pos.self_id),
                pos.self_id, pos.self_index)
        else:
            raise NotImplementedError

    def merge_call(self, state: 'State', return_value: Any) -> None:
        if self.defer_restart is None:  # callee will not perform defer restart
            return
        self.written = True
        self.defer_restart = None
        replacement_mapping: dict[torch.fx.Node, torch.fx.Node] = {}

        def merge_call_guard() -> None:
            for idx, var in state.objects.objs.items():
                if var.need_guard_check:
                    new_var = copy.copy(var)
                    new_var.extract_code_at_start = []
                    for pos in var.extract_code_at_start:
                        if isinstance(pos, StoreInLocal):
                            continue
                        elif isinstance(pos, StoreInStack):
                            raise ValueError(
                                "full graph should not contain guard in stack")
                        elif isinstance(pos, StoreInGlobal):
                            new_var.extract_code_at_start.append(pos)
                        elif isinstance(pos, StoreInAttr):
                            new_var.extract_code_at_start.append(
                                self.store_pos_in_callee(pos, idx))
                        elif isinstance(pos, StoreInIndex):
                            new_var.extract_code_at_start.append(
                                self.store_pos_in_callee(pos, idx))
                        else:
                            raise NotImplementedError(pos)

        def merge_fx_graph() -> None:
            print("to merge", state.fx_graph.result_graph)

            def replacement_fn(node: torch.fx.Node) -> torch.fx.Node:
                return replacement_mapping[node]

            def get_tensor_idx(node: torch.fx.Node) -> int:
                var = node.meta["var"]
                idx = var.idx
                assert isinstance(idx, int) and idx != 0
                return idx

            def get_original_node(node: torch.fx.Node) -> torch.fx.Node:
                idx = get_tensor_idx(node)
                var_in_caller = self.objects.get_by_id(idx)
                assert isinstance(var_in_caller, vs.TensorVar)
                return var_in_caller.fx_node

            for node in state.fx_graph.result_graph.nodes:
                if node.op == "placeholder":
                    replacement_mapping[node] = get_original_node(node)
                    continue
                elif node.op == "output":
                    assert len(node.args) == 1
                    for arg in node.args[0]:
                        tensor_idx = get_tensor_idx(arg)
                        if self.objects.contains_by_id(tensor_idx):
                            continue
                        new_tensor_var = copy.copy(
                            state.objects.get_by_id(tensor_idx))
                        assert isinstance(new_tensor_var, vs.TensorVar)
                        if new_tensor_var.need_guard_check:
                            raise NotImplementedError
                        new_tensor_var.extract_code_at_start = []
                        new_tensor_var.fx_node = replacement_fn(arg)
                        self.objects.add_by_id(new_tensor_var, tensor_idx)
                    continue
                new_node = self.fx_graph.result_graph.node_copy(
                    node, replacement_fn)
                if node.op == "get_attr":
                    param_obj = None
                    for name, obj in state.root.named_parameters():
                        if name == node.target:
                            param_obj = obj
                            break
                    if param_obj is None:
                        raise ValueError(
                            f"cannot find param {node.target} in {state.root}")
                    name_in_caller = self.subparam_paths[param_obj]
                    new_node.target = name_in_caller
                elif node.op == "call_module":
                    module_obj = None
                    for name, obj in state.root.named_modules():
                        if name == node.target:
                            module_obj = obj
                            break
                    if module_obj is None:
                        raise ValueError(
                            f"cannot find module {node.target} in {state.root}")
                    name_in_caller = self.submodule_paths[module_obj]
                    new_node.target = name_in_caller
                elif node.op == "call_method":
                    raise NotImplementedError
                replacement_mapping[node] = new_node

        def merge_output() -> None:
            merged_ids: set[int] = set()

            def get_or_make_var(
                    obj: Any, need_guard_check: bool,
                    fx_graph: Optional[FxGraph],
                    extract_code_at_start: list[StorePos]) -> vs.Variable:
                if id(obj) in merged_ids:
                    new_var = self.objects.get(obj)
                elif self.objects.contains(obj) and self.objects.get(
                        obj, False) is not None:  # value from caller, modified
                    if isinstance(obj, torch.Tensor):
                        old_obj = state.objects.get(obj, False)
                        assert isinstance(old_obj, vs.TensorVar)
                        old_node = old_obj.fx_node
                        new_node = replacement_mapping[old_node]
                        new_var = vs.TensorVar.from_tensor_and_node(
                            obj, new_node, need_guard_check,
                            extract_code_at_start)
                    else:
                        new_var = vs.make_var_from_value(
                            obj, need_guard_check, get_or_make_var, fx_graph,
                            extract_code_at_start)
                    self.objects.update_by_id(new_var, id(obj))
                elif self.objects.contains(
                        obj):  # value from caller, unmodified
                    new_var = self.objects.get(obj)
                else:  # value not in caller
                    if isinstance(obj, torch.Tensor):
                        old_obj = state.objects.get(obj, False)
                        assert isinstance(old_obj, vs.TensorVar)
                        old_node = old_obj.fx_node
                        new_node = replacement_mapping[old_node]
                        new_var = vs.TensorVar.from_tensor_and_node(
                            obj, new_node, need_guard_check,
                            extract_code_at_start)
                    else:
                        new_var = vs.make_var_from_value(
                            obj, need_guard_check, get_or_make_var, fx_graph,
                            extract_code_at_start)
                    self.objects.add(new_var, obj)
                merged_ids.add(id(obj))
                return new_var

            for idx, var in state.objects.get_all_with_id():
                if var.prev is not None:
                    oldest = var.get_oldest_var()
                    if len(oldest.extract_code_at_start) == 0:
                        continue
                    new_extract = [
                        self.store_pos_in_callee(pos, idx)
                        for pos in oldest.extract_code_at_start
                    ]
                    get_or_make_var(var.obj, var.need_guard_check,
                                    self.fx_graph, new_extract)

            var = state.objects.get(return_value, allow_unexist_const=True)
            new_extract = [
                self.store_pos_in_callee(pos, id(var.obj))
                for pos in var.extract_code_at_start
            ]
            get_or_make_var(return_value, var.need_guard_check, self.fx_graph,
                            new_extract)

        if len(state.objects.objs_no_id) > 0:
            raise NotImplementedError
        merge_call_guard()
        merge_fx_graph()
        merge_output()
        self.object_refs.extend(state.object_refs)

    def mark_defer_restart(self, reason: str, stack_objs: list[Any]) -> None:
        self.defer_restart = reason
        self.stack_objs = stack_objs

    def add_inplace_update_obj(self, obj: Any) -> None:
        self.written = True
        self.inplace_update_objs.append(obj)


class GuardTracker:
    code: ProcessedCode
    frame_id: int
    frame: FrameType
    state: State
    have_error: bool
    frame_root: torch.nn.Module
    caller: Optional['GuardTracker']

    def __init__(self,
                 frame: FrameType,
                 frame_id: int,
                 caller: Optional['GuardTracker'] = None):
        self.code = load_code(frame_id)
        self.frame = frame
        self.frame_id = frame_id
        self.frame_root = get_frame_root(frame_id)
        self.caller = caller
        self.init_state(
            read_stack=False
        )  # stack pointer is not initialized at the creation of a stack frame

    def init_state(self, read_stack: bool = True) -> None:
        if hasattr(self, "state"):
            self.state.written = False
        self.state = State.from_frame(self.frame, read_stack, self.frame_root)
        self.have_error = False

    def record(
            self, frame: FrameType, frame_id: int
    ) -> None:  # pass frame and frame_id only for assertion
        # print("frame_id:", frame_id, "self.frame_id:", self.frame_id)
        assert frame_id == self.frame_id
        assert frame == self.frame
        self.process_last_inst()

        pc, inst = self.code.get_orig_inst(self.frame.f_lasti)
        if inst is None:
            self.restart(
                f"running injected code (f_lasti={self.frame.f_lasti})")
            if self.code.get_inst(self.frame.f_lasti).opname == 'RETURN_VALUE':
                if trackers[-1] == self:
                    pop_tracker(self.frame_id)
                set_eval_frame(None)
            return
        if has_force_graph_break(frame_id, pc):
            assert inst.opcode != dis.opmap["LOAD_METHOD"]
            self.restart(f"force graph break (pc = {pc})")
            return
        # call init_state after is_inject_code check to avoid frequent init_state
        if self.have_error:
            try:
                self.init_state()
            except Exception as e:
                self.restart(f"Exception during init: {e}")
                print(traceback.format_exc())
                return
        if self.state.start_pc == -1:
            self.state.start_pc = pc
            assert self.state.start_pc >= 0
        if self.code.get_inst(
                self.frame.f_lasti).opname in ('SETUP_WITH', 'FOR_ITER',
                                               'JUMP_IF_TRUE_OR_POP',
                                               'JUMP_IF_FALSE_OR_POP',
                                               'SETUP_FINALLY', 'RAISE_VARARGS',
                                               'SETUP_ASYNC_WITH'):
            self.state.num_new_refs = -1
        else:
            self.state.num_new_refs = stack_effect(
                inst.opcode,
                inst.arg or 0,
                None,
            )[2]
        if hasattr(self, inst.opname):
            try:
                getattr(self, inst.opname)(inst)
                # NOTE: DO NOT write any function call after this line
                # because frame evaluation function may be set during processing the opcode
            except Exception as e:
                self.restart(f"Exception during processing {inst.opname}: {e}")
            if not self.have_error and self.state.defer_restart is None:
                self.state.is_empty = False
                self.state.written = False
        else:
            self.restart(f"unknown opcode {inst.opname}")

    def commit(self, break_before_cur_inst: bool) -> None:
        assert not self.state.written
        if self.state.is_empty:
            return
        assert self.state.start_pc >= 0
        lasti = self.frame.f_lasti
        if break_before_cur_inst:
            lasti -= 2
        end_pc = self.code.get_orig_pc(lasti)
        if end_pc == -1:
            end_pc = self.code.get_next_orig_pc(lasti)
        print("commiting", self.state.start_pc, end_pc,
              self.code.original_insts[end_pc], lasti)
        key = new_random_key()
        guard_codegen = GuardFnCodegen(key=key)
        for var in self.state.objects.get_all():
            while var.prev is not None:
                var = var.prev
            var.make_guard(guard_codegen)
        guard_code = guard_codegen.get_code()
        graph_codegen = GraphFnCodegen(key=key)
        for node in self.state.fx_graph.result_graph.nodes:
            if node.op == "placeholder":
                var = node.meta["var"]
                assert isinstance(var, vs.TensorVar)
                graph_codegen.add_graph_input(var.extract_code_at_start[0])
        current_inst = self.code.get_inst(lasti)
        # livevars_analysis should return the same result when passing self.code.guard_insts
        # and self.code.original_insts, but as current_inst may not be in original_insts,
        # we pass guard_insts here
        live_vars = livevars_analysis(self.code.guard_insts, current_inst)
        live_vars = live_vars.intersection(self.state.stored_locals)
        for i, live_var in enumerate(live_vars):
            value = self.frame.f_locals[live_var]
            var = self.state.objects.get(value, allow_unexist_const=True)
            var.make_output(f"__live_{i}", StoreInLocal(live_var),
                            graph_codegen, True, id(value))
        # TODO: can be optimized by only reproduce the modified variables
        if break_before_cur_inst:
            stack_objs = self.state.stack_objs
            assert stack_objs is not None
        else:
            stack_objs = get_all_objects_in_stack(self.frame)

        for i, value in enumerate(stack_objs):
            var = self.state.objects.get(value, allow_unexist_const=True)
            var.make_output(f"__stack__{i}", StoreInStack(i), graph_codegen,
                            True, id(value))

        for idx, var in self.state.objects.get_all_with_id():
            if var.prev is not None and idx not in graph_codegen.id2name:
                oldest_var = var.get_oldest_var()
                if len(oldest_var.extract_code_at_start) == 0:
                    continue
                var.make_output(f"__tmp_{idx}",
                                oldest_var.extract_code_at_start[0],
                                graph_codegen, False, idx)

        self.state.fx_graph.set_output_nodes(graph_codegen.get_graph_outputs())
        print("graph", self.state.fx_graph.result_graph)

        if self.state.start_pc == 0 and self.code.original_insts[
                end_pc].opname == "RETURN_VALUE" and self.caller is not None:
            print("callee is full graph, merge to caller")
            assert len(stack_objs) == 1
            self.caller.state.merge_call(
                self.state, get_value_stack_from_top(self.frame, 0))
        else:
            graph_code = graph_codegen.get_code()
            compiled_graph = self.state.fx_graph.compile()

            py_code = f"""\
{graph_code}
{guard_code}
            """
            out: Dict[str, Any] = dict()
            print("RUNNING PY CODE")
            print(py_code)
            exec(py_code, self.frame.f_globals, out)
            guard_fn = out["___make_guard_fn"](*guard_codegen.vars.values())
            graph_fn = out["___make_graph_fn"](compiled_graph,
                                               *graph_codegen.objs.values())

            print("guard_fn:", guard_fn)
            print("pc:", self.state.start_pc, end_pc)
            print("stack:", self.state.start_stack_size, len(stack_objs))

            get_frame_cache(self.frame_id).add(
                CachedGraph(
                    guard_fn,
                    graph_fn,
                    self.state.start_pc,
                    end_pc,
                    start_stack_size=self.state.start_stack_size,
                    end_stack_size=len(stack_objs),
                    return_values=graph_codegen.get_return_values(),
                    key=key,
                    object_refs=guard_codegen.get_object_refs(),
                ))

        self.state.is_empty = True

    def process_last_inst(self) -> None:
        if self.state.num_new_refs == -1:
            self.state.num_new_refs = get_value_stack_size(self.frame)
        for i in range(self.state.num_new_refs):
            self.state.object_refs.append(
                get_value_stack_from_top(self.frame, i))
        self.state.num_new_refs = 0
        if len(self.state.inplace_update_objs) > 0:
            for i, obj in enumerate(self.state.inplace_update_objs):
                if isinstance(obj, StoreInStack):
                    pos = obj
                    obj = get_value_stack_from_top(self.frame, pos.idx)
                    if isinstance(obj, torch.Tensor):
                        new_var: vs.Variable = vs.TensorVar.from_tensor_and_node(
                            obj, self.state.node_waiting_ids[pos.idx], False)
                    else:
                        new_var = vs.make_var_from_value(
                            obj, False, self.state.objects.get_or_make_var,
                            self.state.fx_graph, [])
                else:
                    # heheda: not know how to get the node of tensor now
                    assert not isinstance(obj, torch.Tensor)
                    new_var = vs.make_var_from_value(
                        obj, False, self.state.objects.get_or_make_var,
                        self.state.fx_graph, [])
                self.state.objects.update_by_id(new_var, id(obj))
        else:
            for i, node in enumerate(self.state.node_waiting_ids):
                value = get_value_stack_from_top(self.frame, i)
                if isinstance(value, torch.Tensor):
                    var = vs.TensorVar.from_tensor_and_node(value, node, False)
                else:
                    raise NotImplementedError
                self.state.objects.add(var, value)
        self.state.inplace_update_objs.clear()
        self.state.node_waiting_ids.clear()
        if self.state.defer_restart is not None:
            self.restart("defered restart " + self.state.defer_restart,
                         break_before_cur_inst=True)

    def restart(self,
                restart_reason: str,
                break_before_cur_inst: bool = False) -> None:
        logging.info(f"restart: {restart_reason}")
        self.have_error = True
        self.commit(break_before_cur_inst)

    @classmethod
    def has_tensor_arg(cls, args: List[Any], kwargs: Dict[str, Any]) -> bool:
        return any(
            isinstance(i, torch.Tensor)
            for i in itertools.chain(args, kwargs.values()))

    @classmethod
    def all_scalar_arg(cls, args: List[Any], kwargs: Dict[str, Any]) -> bool:
        return all(is_scalar(i) for i in itertools.chain(args, kwargs.values()))

    @classmethod
    def has_tuple_arg(cls, args: List[Any], kwargs: Dict[str, Any]) -> bool:
        return any(
            isinstance(i, tuple)
            for i in itertools.chain(args, kwargs.values()))

    @classmethod
    def has_list_arg(cls, args: List[Any], kwargs: Dict[str, Any]) -> bool:
        return any(
            isinstance(i, list) for i in itertools.chain(args, kwargs.values()))

    def call_function(
        self,
        func: Callable[..., Any],
        args: List[Any],
        kwargs: Dict[str, Any],
    ) -> None:
        if isinstance(
                func,
                torch.nn.Module) and func not in self.state.submodule_paths:
            self.state.add_submodule(func)
        if is_user_defined_func(func):
            stack_objs = get_all_objects_in_stack(self.frame)
            self.state.mark_defer_restart(f"call_function {func}", stack_objs)
            from .tracer import get_process_frame
            preprocess_frame, post_process_frame = get_process_frame(func, True)
            prior = set_eval_frame((preprocess_frame, post_process_frame))
            assert prior is None
            assert self.state.written == False
            return
        if func in fx_graph_inplace_functions:
            if len(args) == 0:
                raise NotImplementedError
            if not self.state.objects.contains(args[0]):
                self.state.add_object(
                    vs.make_var_from_value(args[0], False,
                                           self.state.objects.get_or_make_var,
                                           self.state.fx_graph, []), args[0])
            self.state.inplace_update_objs.append(StoreInStack(0))
        if self.has_tensor_arg(args, kwargs):
            if func in fx_graph_functions or isinstance(func, torch.nn.Module):
                self.state.record_function(func, args, kwargs)
                return
            func_var = self.state.objects.get_or_none(func)
            if func_var is not None:
                assert isinstance(func_var, vs.FunctionVar)
                if func_var.src == vs.ObjectSrc.TORCH:
                    self.state.record_function(func, args, kwargs)
                    return

        elif self.all_scalar_arg(args, kwargs):
            return
        elif self.has_tuple_arg(args, kwargs):
            return
        elif self.has_list_arg(args, kwargs):
            return
        raise NotImplementedError

    def binary_operation(self, func: Callable[..., Any]) -> None:
        obj1 = get_value_stack_from_top(self.frame, 1)
        obj2 = get_value_stack_from_top(self.frame, 0)
        self.call_function(func, [obj1, obj2], {})

    def BINARY_ADD(self, _inst: Instruction) -> None:
        self.binary_operation(operator.add)

    def BINARY_SUBTRACT(self, _inst: Instruction) -> None:
        self.binary_operation(operator.sub)

    def BINARY_MULTIPLY(self, _inst: Instruction) -> None:
        self.binary_operation(operator.mul)

    def BINARY_FLOOR_DIVIDE(self, _inst: Instruction) -> None:
        self.binary_operation(operator.floordiv)

    def BINARY_TRUE_DIVIDE(self, _inst: Instruction) -> None:
        self.binary_operation(operator.truediv)

    def BINARY_MODULO(self, _inst: Instruction) -> None:
        self.binary_operation(operator.mod)

    def BINARY_POWER(self, _inst: Instruction) -> None:
        self.binary_operation(operator.pow)

    def BINARY_LSHIFT(self, _inst: Instruction) -> None:
        self.binary_operation(operator.lshift)

    def BINARY_RSHIFT(self, _inst: Instruction) -> None:
        self.binary_operation(operator.rshift)

    def BINARY_AND(self, _inst: Instruction) -> None:
        self.binary_operation(operator.and_)

    def BINARY_XOR(self, _inst: Instruction) -> None:
        self.binary_operation(operator.xor)

    def BINARY_OR(self, _inst: Instruction) -> None:
        self.binary_operation(operator.or_)

    def BINARY_SUBSCR(self, inst: Instruction) -> None:
        obj1 = get_value_stack_from_top(self.frame, 1)
        obj2 = get_value_stack_from_top(self.frame, 0)
        self.call_function(operator.getitem, [obj1, obj2], {})

    def INPLACE_POWER(self, _inst: Instruction) -> None:
        self.binary_operation(operator.ipow)

    def INPLACE_MULTIPLY(self, _inst: Instruction) -> None:
        self.binary_operation(operator.imul)

    def INPLACE_MATRIX_MULTIPLY(self, _inst: Instruction) -> None:
        self.binary_operation(operator.imatmul)

    def INPLACE_FLOOR_DIVIDE(self, _inst: Instruction) -> None:
        self.binary_operation(operator.ifloordiv)

    def INPLACE_TRUE_DIVIDE(self, _inst: Instruction) -> None:
        self.binary_operation(operator.itruediv)

    def INPLACE_MODULO(self, _inst: Instruction) -> None:
        self.binary_operation(operator.imod)

    def INPLACE_ADD(self, _inst: Instruction) -> None:
        self.binary_operation(operator.iadd)

    def INPLACE_SUBTRACT(self, _inst: Instruction) -> None:
        self.binary_operation(operator.isub)

    def INPLACE_LSHIFT(self, _inst: Instruction) -> None:
        self.binary_operation(operator.ilshift)

    def INPLACE_RSHIFT(self, _inst: Instruction) -> None:
        self.binary_operation(operator.irshift)

    def INPLACE_AND(self, _inst: Instruction) -> None:
        self.binary_operation(operator.iand)

    def INPLACE_XOR(self, _inst: Instruction) -> None:
        self.binary_operation(operator.ixor)

    def INPLACE_OR(self, _inst: Instruction) -> None:
        self.binary_operation(operator.ior)

    def BUILD_SLICE(self, _inst: Instruction) -> None:
        pass

    def LOAD_CONST(self, _inst: Instruction) -> None:
        pass

    def LOAD_FAST(self, inst: Instruction) -> None:
        if inst.argval not in self.state.stored_locals:
            obj = self.frame.f_locals[inst.argval]
            pos = StoreInLocal(inst.argval)
            if not self.state.objects.contains(obj):
                var = vs.make_var_from_value(obj, True,
                                             self.state.objects.get_or_make_var,
                                             self.state.fx_graph, [pos])
                self.state.add_object(var, obj)
                print("add to obj table", obj, id(obj))
            else:
                var = self.state.objects.get(obj)
                var.extract_code_at_start.append(pos)

    def LOAD_GLOBAL(self, inst: Instruction) -> None:
        if inst.argval not in self.state.stored_globals:
            try:
                if inst.argval in self.frame.f_globals:
                    obj = self.frame.f_globals[inst.argval]
                else:  # try first search in __builtins__
                    obj = getattr(self.frame.f_globals['__builtins__'],
                                  str(inst.argval))
            except Exception as e:
                raise UnknownTypeError(inst.argval)

            var = vs.make_var_from_value(obj, True,
                                         self.state.objects.get_or_make_var,
                                         self.state.fx_graph,
                                         [StoreInGlobal(inst.argval)])
            self.state.add_object(var, obj)

    # heheda: we need to make sure that no unbound LOAD_METHOD is called by python runtime to avoid NULL in stack
    def LOAD_METHOD(self, inst: Instruction) -> None:
        self_obj = get_value_stack_from_top(self.frame, 0)
        method = getattr(self_obj, inst.argval)
        self_var = self.state.objects.get(self_obj)
        method_var = vs.make_var_from_value(
            method, self_var.need_guard_check,
            self.state.objects.get_or_make_var, self.state.fx_graph, [
                StoreInAttr(self_var.extract_code_at_start[0], id(self_obj),
                            inst.argval)
            ] if self_var.need_guard_check else [])
        if isinstance(self_var, vs.ModuleVar) and isinstance(
                method_var, vs.FunctionVar):
            method_var.src = self_var.src
        self.state.add_object(method_var, method)

    def LOAD_ATTR(self, inst: Instruction) -> None:
        obj = get_value_stack_from_top(self.frame, 0)
        attr = getattr(obj, inst.argval)
        obj_var = self.state.objects.get(obj)
        attr_var = vs.make_var_from_value(
            attr, obj_var.need_guard_check, self.state.objects.get_or_make_var,
            self.state.fx_graph, [
                StoreInAttr(obj_var.extract_code_at_start[0], id(obj),
                            inst.argval)
            ] if obj_var.need_guard_check else [])
        if isinstance(obj_var, vs.ModuleVar):
            if isinstance(attr_var, vs.ModuleVar):
                attr_var.src = obj_var.src
        self.state.add_object(attr_var, attr)

    def CALL_FUNCTION(self, inst: Instruction) -> None:
        num_args = inst.argval
        args = [
            get_value_stack_from_top(self.frame, i)
            for i in range(num_args - 1, -1, -1)
        ]
        kwargs: dict[str, Any] = {}
        func = get_value_stack_from_top(self.frame, num_args)
        self.call_function(func, args, kwargs)

    def CALL_METHOD(self, inst: Instruction) -> None:
        num_args = inst.argval
        args = [
            get_value_stack_from_top(self.frame, i)
            for i in range(num_args - 1, -1, -1)
        ]
        kwargs: dict[str, Any] = {}
        self_val = get_value_stack_from_top(self.frame, num_args)
        meth_val = get_value_stack_from_top(self.frame, num_args + 1)
        if isinstance(meth_val, NullObject):
            # Stack layout: ... | NULL | callable | arg1 | ... | argN
            self.call_function(self_val, args, kwargs)
        else:
            # Stack layout: ... | method | self | arg1 | ... | argN
            self.call_function(meth_val, [self_val] + args, kwargs)

    def CALL_FUNCTION_KW(self, inst: Instruction) -> None:
        num_args = inst.argval
        args = [
            get_value_stack_from_top(self.frame, i + 1)
            for i in range(num_args - 1, -1, -1)
        ]
        kw_names = get_value_stack_from_top(self.frame, 0)
        func = get_value_stack_from_top(self.frame, num_args + 2)
        kwargs: dict[str, Any] = {}
        for arg, kw_name in zip(args[-len(kw_names):], kw_names):
            kwargs[kw_name] = arg
        args = args[:-len(kw_names)]
        self.call_function(func, args, kwargs)

    '''
    not tested due to lack of dict and list type
    def CALL_FUNCTION_EX(self, inst: Instruction) -> None:
        offset = inst.argval & 1
        func = get_value_stack_from_top(self.frame, 1 + offset)
        args = get_value_stack_from_top(self.frame, offset)
        if offset == 1:
            kwargs = get_value_stack_from_top(self.frame, 0)
        else:
            kwargs = {}
        self.call_function(func, args, kwargs)
    '''

    def STORE_FAST(self, inst: Instruction) -> None:
        self.state.add_stored_locals(inst.argval)

    def STORE_SUBSCR(self, inst: Instruction) -> None:
        index = get_value_stack_from_top(self.frame, 0)
        target = get_value_stack_from_top(self.frame, 1)
        value = get_value_stack_from_top(self.frame, 2)
        if isinstance(target, torch.Tensor):
            # still use the original node, so no need to update object table
            self.state.record_function(operator.setitem, [target, index, value],
                                       {},
                                       add_proxy_waiting_id=False)
        else:
            self.state.add_inplace_update_obj(target)

    def BUILD_TUPLE(self, inst: Instruction) -> None:
        pass

    def BUILD_LIST(self, inst: Instruction) -> None:
        pass

    def BUILD_SET(self, inst: Instruction) -> None:
        pass

    # def LIST_TO_TUPLE(self, inst: Instruction) -> None:
    #     pass

    def LIST_EXTEND(self, inst: Instruction) -> None:
        pass

    def POP_TOP(self, _inst: Instruction) -> None:
        pass

    def ROT_TWO(self, _inst: Instruction) -> None:
        pass

    def ROT_THREE(self, _inst: Instruction) -> None:
        pass

    def ROT_FOUR(self, _inst: Instruction) -> None:
        pass

    def DUP_TOP(self, _inst: Instruction) -> None:
        pass

    def DUP_TOP_TWO(self, _inst: Instruction) -> None:
        pass

    def POP_JUMP_IF_FALSE(self, _inst: Instruction) -> None:
        pass

    def POP_JUMP_IF_TRUE(self, _inst: Instruction) -> None:
        pass

    def JUMP_IF_TRUE_OR_POP(self, _inst: Instruction) -> None:
        pass

    def JUMP_IF_FALSE_OR_POP(self, _inst: Instruction) -> None:
        pass

    def JUMP_FORWARD(self, inst: Instruction) -> None:
        pass

    def JUMP_ABSOLUTE(self, inst: Instruction) -> None:
        pass


trackers: list[GuardTracker] = []


def push_tracker(frame: FrameType, frame_id: int) -> None:
    if len(trackers) > 0:
        caller = trackers[-1]
    else:
        caller = None
    trackers.append(GuardTracker(frame, frame_id, caller))
    print("push tracker", frame_id, "frame", hex(id(frame)), "frame_id",
          frame_id, "all", [t.frame_id for t in trackers])


def pop_tracker(frame_id: int) -> None:
    print("before pop_tracker", [t.frame_id for t in trackers], "frame_id",
          frame_id)
    to_pop = trackers.pop()
    assert to_pop.frame_id == frame_id
    assert to_pop.state.is_empty


def record(frame: FrameType, frame_id: int) -> None:
    if frame_id != trackers[-1].frame_id:
        last_inst = trackers[-1].code.get_inst(trackers[-1].frame.f_lasti)
        if is_call_bytecode(last_inst):
            push_tracker(frame, frame_id)
    trackers[-1].record(frame, frame_id)


def reset() -> None:
    trackers.clear()
