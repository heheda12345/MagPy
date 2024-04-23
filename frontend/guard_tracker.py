from types import FrameType, MappingProxyType, ModuleType
from typing import Dict, Any, Callable, List, Optional, cast, Union
import inspect
import logging
import itertools
import torch
import torch.nn as nn
import torch.fx
import operator
import dis
import traceback
import copy
import dataclasses
import collections
import torch.fx.immutable_collections as fx_immutable
import numpy as np
from . import config
from .code import ProcessedCode
from .c_api import get_value_stack_from_top, get_value_stack_size, set_eval_frame, stack_effect, get_code_map, is_bound_method, get_from_freevars, set_value_stack_from_top, parse_cell, set_local
from .instruction import Instruction, ci
from .cache import CachedGraph, get_frame_cache
from .store_pos import StoreConstant, StorePos, StoreInStack, StoreInLocal, StoreInGlobal, StoreInAttr, StoreInIndex, ExtractFromMethod, StoreInBuiltin, ExtractFromFunction, IterValue, StoreInFreeVar, ExtractFromNew, UnknownPosInCaller
from . import variables as vs
from . import dynamic as dyn
from .utils import is_scalar, new_random_key, has_force_graph_break, NullObject, is_call_bytecode, fx_graph_functions, fx_graph_inplace_functions, is_user_defined_func, UnknownTypeError, get_all_objects_in_stack, is_graph_func, get_root_module, torch_inplace_funcs, print_bytecode, get_method_defined_class, is_math_func, is_high_order_func_with_udf, is_high_order_func, math2torch
from .object_table import ObjectTable
from .pycode_writer import new_name
from .pycode_generator import GraphFnCodegen, GuardFnCodegen
from .fx_graph import FxGraph, get_frame_root, is_leaf_module, NodeArgs
from .bytecode_analysis import livevars_analysis, end_of_control_flow
from .variables.const import ClsByNamedTupleVar
from .variables.base import Variable
from .control_flow import ControlFlowInfo, LoopModule, ForLoopInfo, LoopPosMap, if_stmt, IfStmtInfo

MAKE_VAR_FN_TYPE = Callable[[
    Any, bool, vs.HelperFunctions, Optional[FxGraph], Optional[list[StorePos]]
], Variable]


@dataclasses.dataclass
class PartialVar:
    node: Optional[torch.fx.Node]
    need_guard_check: bool
    extract_code_at_start: list[StorePos]
    inplace_ref: Any = None  # None if not inplace
    make_var_fn: Optional[MAKE_VAR_FN_TYPE] = None
    force_new_value: bool = False
    named_class: bool = False
    named_func: bool = False


@dataclasses.dataclass
class DeferRestartState:
    stack_objs: list[Any]
    live_vars: list[tuple[str, Any]]
    guard_lasti: int
    reason: str
    need_restart: bool


deberta_model = None

the_first_input = None


class State:
    objects: ObjectTable
    start_pc: int
    start_stack_size: int
    is_empty: bool
    fx_graph: FxGraph
    root: torch.nn.Module
    partial_var: dict[int, list[Optional[
        PartialVar]]]  # None for placeholders,key is guarded pc, -1 for any pc
    stored_locals: set[str]
    stored_globals: set[str]
    submodule_paths: dict[torch.nn.Module, str]
    subparam_paths: dict[torch.nn.Parameter, str]
    written: bool
    defer_restart: Optional[DeferRestartState]  # None if no need to restart
    object_refs: list[Any]  # hold the reference of objects to avoid GC
    inplace_update_objs: list[Any]
    guarded_pcs: list[int]
    initial_args: list[Any]
    varargs: Optional[Any]
    varkw: Optional[Any]
    calling_func: Optional[Callable[..., Any]]
    callee_returns: Any
    can_guard: bool
    gen_by_caller: Callable[[Any], bool]
    frame_id: int
    frame_cf_info: Optional[ControlFlowInfo]
    named_funcs: list[ClsByNamedTupleVar]

    def __init__(self, root: torch.nn.Module,
                 gen_by_caller: Callable[[Any], bool]) -> None:
        self.gen_by_caller = gen_by_caller
        self.objects = ObjectTable(self.gen_by_caller, self.mark_cannot_guard)
        self.start_pc = -1
        self.start_stack_size = -1
        self.is_empty = True

        def get_mark_written_fn(state: 'State') -> Callable[[], None]:
            return lambda: setattr(state, "written", True)

        self.fx_graph = FxGraph(root, get_mark_written_fn(self))
        self.root = root
        self.partial_var = {}
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
        self.guarded_pcs = []
        self.initial_args = []
        self.varargs = None
        self.varkw = None
        self.calling_func = None
        self.can_guard = True
        self.frame_id = -1
        self.callee_returns = None
        self.frame_cf_info = None
        self.named_funcs = []

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
        new_module_name = "__external_module__" + str(len(self.submodule_paths))
        self.root.add_module(new_module_name, module)
        self.update_subpath(module, new_module_name)
        # self.written = True # not mark as written as graph break may happen

    def add_subparam(self, param: torch.nn.Parameter) -> str:
        new_param_name = "external_param__" + str(len(self.subparam_paths))
        self.root.register_parameter(new_param_name, param)
        self.subparam_paths[param] = new_param_name
        return new_param_name

    def as_node_args_kwargs(
        self, args: list[Any], kwargs: dict[str, Any]
    ) -> tuple[tuple[torch.fx.Node, ...], dict[str, torch.fx.Node]]:
        common_device: torch.device = None

        def get_common_device(arg: Any) -> None:
            if isinstance(arg, (tuple, list)):
                for a in arg:
                    get_common_device(a)
            if isinstance(arg, torch.Tensor):
                # var = self.objects.get(arg, allow_unexist_const=True)
                nonlocal common_device
                # assert common_device is None or common_device == var.obj.device or var.obj.dim(
                # ) <= 1
                common_device = arg.device

        def as_fx_node(arg: Any) -> NodeArgs:
            if isinstance(arg, (tuple, list)):
                if isinstance(arg, list):
                    return fx_immutable.immutable_list(
                        [as_fx_node(x) for x in arg])
                else:
                    return tuple(
                        fx_immutable.immutable_list(
                            [as_fx_node(x) for x in arg]))
            if isinstance(arg, slice):
                return slice(as_fx_node(arg.start), as_fx_node(arg.stop),
                             as_fx_node(arg.step))
            if isinstance(arg, np.ndarray):
                param_name = self.add_subparam(
                    torch.nn.Parameter(torch.tensor(arg), requires_grad=False))
                return self.fx_graph.create_node("get_attr", param_name, (), {})

            var = self.objects.get(arg,
                                   allow_unexist_const=True,
                                   fx_graph=self.fx_graph)
            if isinstance(var, vs.TorchParamVar):
                if var.obj not in self.subparam_paths:
                    self.add_subparam(var.obj)
                return self.fx_graph.create_node("get_attr",
                                                 self.subparam_paths[var.obj],
                                                 (), {})
            if isinstance(var, vs.ScalarVar) and not var.value_fix:
                if not config.get_config("dynshape"):
                    if common_device is not None and common_device != torch.device(
                            'cpu'):
                        cpu_node = var.as_fx_node()
                        # return self.fx_graph.create_node(
                        #     "call_method", "to", (cpu_node,),
                        #     {"device": common_device})
                        return cpu_node
                else:
                    # TODO: record all operation in SymInt or SymFloat
                    pass
            if isinstance(var, vs.FunctionVar):
                if hasattr(var.obj, '__name__') and hasattr(
                        var.obj, "__module__"):
                    assert var.obj.__module__ in ('torch', 'numpy')
                    return f'{var.obj.__module__}.{var.obj.__name__}'

            if f"{type(arg).__module__}.{type(arg).__qualname__}" == "torch.tensortype":  # torch.LongTensor
                return f"torch.{arg.__name__}"
            return var.as_fx_node()

        if isinstance(args, torch.Tensor):
            args_list = []
            arg_fx_node = as_fx_node(args)

            for i, arg in enumerate(args):
                fx_node = self.fx_graph.create_node(
                    "call_function",
                    operator.getitem,
                    (arg_fx_node, i),
                    {},
                )
                var = vs.TensorVar.from_tensor_and_node(arg, fx_node, False, [])
                self.objects.add(var, arg)
                fx_node.meta["var"] = var
                args_list.append(arg)
            args = args_list

        for arg in itertools.chain(args, kwargs.values()):
            get_common_device(arg)
        node_args = tuple(as_fx_node(arg) for arg in args)
        node_kwargs = {key: as_fx_node(arg) for key, arg in kwargs.items()}

        return node_args, node_kwargs

    def record_function(self,
                        func: Callable[..., Any],
                        args: List[Any],
                        kwargs: Dict[str, Any],
                        add_partial_var: bool = True,
                        inplace_ref: Any = None,
                        force_new_value: bool = False) -> None:
        if hasattr(func, '__self__') and isinstance(
                func.__self__, torch.autograd.grad_mode.no_grad):
            if func.__name__ == '__enter__':
                target_state = False
            elif func.__name__ == '__exit__':
                target_state = func.__self__.prev
            elif func.__name__ == 'clone':
                target_state = False
            else:
                raise ValueError(func)
            args = [
                target_state,
            ]
            func = torch._C._set_grad_enabled
            kwargs = {}
        pargs, pkwargs = self.as_node_args_kwargs(args, kwargs)
        if func in fx_graph_inplace_functions:
            scalar = None
            node = None
            for i, obj in enumerate(pargs):
                if isinstance(obj, (int, float)) and not dyn.contains(obj):
                    scalar = obj
                    position = i
                else:
                    node = obj
            if scalar is not None and node is not None and position == 0:
                fx_node = self.fx_graph.create_node(
                    "call_function",
                    torch.full_like,
                    (node, scalar),
                    {},
                )
                pargs = (fx_node, node)
        if func in (min, max):
            scalar = None
            node = None
            # NOTE: when pargs < 2, it should be a dynamic operation
            assert len(pargs) <= 2
            for i, obj in enumerate(pargs):
                if isinstance(obj, (int, float)) and not dyn.contains(obj):
                    scalar = obj
                    position = i
                else:
                    node = obj
            if scalar is not None and node is not None and not config.get_config(
                    'dynshape'):
                fx_node = self.fx_graph.create_node(
                    "call_function",
                    torch.full_like,
                    (node, scalar),
                    {},
                )
                pargs_list = list(pargs)
                pargs_list[position] = fx_node
                pargs = tuple(pargs_list)
                func_dict = {min: torch.minimum, max: torch.maximum}
                func = func_dict[func]
        if func in math2torch:
            func = math2torch[func]
        if func == torch.from_numpy:
            func = torch.tensor
        if hasattr(func, '__name__') and func.__name__ == 'numpy':
            if torch.is_tensor(args[0]) or dyn.contains(args[0]):
                raise ValueError("numpy can't have dynamic args")
        self.written = True
        scalar2tensor: dict[Callable[..., Any], Callable[..., Any]] = {
            float: torch.Tensor.float,
            int: torch.Tensor.long,
        }
        if isinstance(func, torch.nn.Module):
            if is_leaf_module(func):
                fx_node = self.fx_graph.create_node(
                    "call_module",
                    self.submodule_paths[func],
                    pargs,
                    pkwargs,
                )
                if add_partial_var:
                    self.partial_var = {
                        -1: [
                            PartialVar(node=fx_node,
                                       need_guard_check=False,
                                       extract_code_at_start=[],
                                       inplace_ref=inplace_ref,
                                       force_new_value=force_new_value)
                        ]
                    }
            else:
                for k, v in self.submodule_paths.items():
                    print(id(k), v, k)
                raise NotImplementedError(func)
        elif (hasattr(func, '__self__') and
              isinstance(func.__self__, torch.Tensor)) or (
                  hasattr(func, '__objclass__') and func.__objclass__
                  == torch._C._TensorBase) or func in scalar2tensor:
            if func in scalar2tensor:
                func = scalar2tensor[func]
            elif func == torch.Tensor.new and len(args) >= 2 and isinstance(
                    args[1], torch.Size):
                func = torch.Tensor.new_empty
            elif func == torch.Tensor.item:
                assert args[0].numel() == 1
                if args[0].dtype == torch.bool:
                    raise ValueError(
                        "The .item() method was applied to a boolean tensor.")
                func = torch.Tensor.clone

            fx_node = self.fx_graph.create_node("call_method", func.__name__,
                                                pargs, pkwargs)
            if func.__name__ == 'tolist':
                add_partial_var = False
            if add_partial_var:
                self.partial_var = {
                    -1: [
                        PartialVar(node=fx_node,
                                   need_guard_check=False,
                                   extract_code_at_start=[],
                                   inplace_ref=inplace_ref,
                                   force_new_value=force_new_value)
                    ]
                }
        else:
            fx_node = self.fx_graph.create_node(
                "call_function",
                func,
                pargs,
                pkwargs,
            )
            if add_partial_var:
                self.partial_var = {
                    -1: [
                        PartialVar(node=fx_node,
                                   need_guard_check=False,
                                   extract_code_at_start=[],
                                   inplace_ref=inplace_ref,
                                   force_new_value=force_new_value)
                    ]
                }

    @classmethod
    def from_frame(cls, frame: FrameType, frame_id: int, read_stack: bool,
                   frame_root: torch.nn.Module, gen_by_caller: Callable[[Any],
                                                                        bool],
                   frame_cf_info: Optional[ControlFlowInfo]) -> 'State':
        state = cls(frame_root, gen_by_caller)
        if read_stack:
            state.start_stack_size = get_value_stack_size(frame)
            for i in range(state.start_stack_size):
                value = get_value_stack_from_top(frame, i)
                var = vs.make_var_from_value(value, True,
                                             state.objects.helper_functions,
                                             state.fx_graph,
                                             [StoreInLocal(f"__stack__{i}")])
                state.objects.add(var, value)
        f_code = frame.f_code
        # state.written may be assigned inside make_var_from_value
        nargs = f_code.co_argcount + f_code.co_kwonlyargcount
        for var_name in frame.f_code.co_varnames[:nargs]:
            state.initial_args.append(frame.f_locals[var_name])
        CO_VARARGS = 0x4
        if f_code.co_flags & CO_VARARGS:
            var_name = f_code.co_varnames[nargs]
            nargs += 1
            state.varargs = frame.f_locals[var_name]
        CO_VARKEYWORDS = 0x8
        if f_code.co_flags & CO_VARKEYWORDS:
            var_name = f_code.co_varnames[nargs]
            nargs += 1
            state.varkw = frame.f_locals[var_name]

        state.written = False
        state.frame_id = frame_id
        state.frame_cf_info = frame_cf_info
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

    def delete_stored_locals(self, name: str) -> None:
        self.written = True
        self.stored_locals.remove(name)

    def store_pos_in_caller(self, pos: StorePos,
                            idx: int) -> Optional[StorePos]:
        if idx in self.objects.objs:
            var = self.objects.objs[idx]
            if len(var.extract_code_at_start) == 0:
                return None
            return var.extract_code_at_start[0]
        elif hasattr(pos, "self_id"):
            import _ctypes
            if hasattr(_ctypes, "PyObj_FromPtr"):
                obj = _ctypes.PyObj_FromPtr(pos.self_id)
            if isinstance(obj, tuple):
                for i in obj:
                    if isinstance(i, torch.Tensor) and self.objects.contains(i):
                        continue
                    elif is_scalar(i):
                        continue
                    else:
                        raise ValueError(f"unknow value in merged tuple: {i}")
                tuple_var = vs.tuple_.TupleVar.from_value(
                    obj, False, self.objects.helper_functions, self.fx_graph,
                    [])
                self.objects.add(tuple_var, obj)
                return None
        if isinstance(pos, StoreInLocal):
            raise ValueError("unknown local in callee", pos)
        elif isinstance(pos, StoreInStack):
            raise ValueError("cannot store in stack in callee")
        elif isinstance(pos, (StoreInGlobal, StoreInBuiltin, StoreInFreeVar)):
            return pos
        elif isinstance(pos, StoreConstant):
            return pos
        elif isinstance(pos, StoreInAttr):
            # print("in callee", pos, self.frame_id)
            parent_pos = self.store_pos_in_caller(pos.self_pos, pos.self_id)
            if parent_pos is None:
                return None
            return StoreInAttr(parent_pos, pos.self_id, pos.attr_name)
        elif isinstance(pos, StoreInIndex):
            parent_pos = self.store_pos_in_caller(pos.self_pos, pos.self_id)
            if parent_pos is None:
                return None
            return StoreInIndex(parent_pos, pos.self_id, pos.self_index)
        elif isinstance(pos, ExtractFromMethod):
            parent_pos = self.store_pos_in_caller(pos.self_pos, pos.self_id)
            if parent_pos is None:
                return None
            return ExtractFromMethod(parent_pos, pos.self_id, pos.method_name)
        elif isinstance(pos, ExtractFromFunction):
            parent_poses: list[StorePos] = []
            for p, i in zip(pos.var_pos, pos.var_id):
                new_pos = self.store_pos_in_caller(p, i)
                if new_pos is None:
                    if isinstance(
                            p,
                            StoreConstant):  # allow constant function parameter
                        new_pos = p
                    else:
                        return None
                parent_poses.append(new_pos)
            return ExtractFromFunction(parent_poses, pos.var_id, pos.func_name,
                                       pos.func_obj, pos.need_add_to_fn)
        elif isinstance(pos, ExtractFromNew):
            return None
        else:
            # print("pos", pos, idx, self.frame_id)
            raise NotImplementedError

    def merge_call(self, state: 'State', stack_objs: list[Any]) -> None:
        print("to merge graph", state.fx_graph.result_graph)
        print("to merge frameid", state.frame_id, self.frame_id)
        # self.written = True
        # self.defer_restart = None
        replacement_mapping: dict[torch.fx.Node, torch.fx.Node] = {}
        calling_func = self.calling_func
        ignore_objs: list[Any] = []
        if calling_func is not None:
            if inspect.isclass(
                    calling_func) and not is_high_order_func(calling_func):
                new_obj = state.initial_args[0]
                assert isinstance(new_obj, calling_func)
                assert not self.objects.contains(new_obj)
                self.objects.add(
                    vs.make_var_from_value(new_obj, False,
                                           self.objects.helper_functions,
                                           self.fx_graph,
                                           [ExtractFromNew(calling_func)]),
                    new_obj)
            if hasattr(calling_func, '__name__') and \
                calling_func.__name__ == 'apply' and \
                hasattr(calling_func,'__self__') and \
                get_method_defined_class(calling_func.__self__, calling_func.__name__) == torch.autograd.function.Function:
                # autograd_function's apply method
                for arg_id in [0, 1]:
                    if len(state.initial_args) > arg_id and isinstance(
                            state.initial_args[arg_id],
                            torch.autograd.function.BackwardCFunction):
                        ignore_objs.append(state.initial_args[arg_id])
                        break
        ignore_ids = [id(x) for x in ignore_objs]
        obj_id_need_merge: set[int] = set()

        def merge_call_guard() -> None:
            for idx, var in state.objects.objs.items():
                if var.need_guard_check:
                    new_var = copy.copy(var)
                    new_var.clear_extract_code_at_start()
                    for pos in var.extract_code_at_start:
                        if isinstance(pos, (StoreInLocal, StoreInFreeVar)):
                            continue
                        elif isinstance(pos, StoreInStack):
                            raise ValueError(
                                "full graph should not contain guard in stack")
                        elif isinstance(pos, (StoreInGlobal, StoreInBuiltin)):
                            new_var.add_extract_code_at_start(pos)
                        elif isinstance(
                                pos, (StoreInAttr, StoreInIndex,
                                      ExtractFromMethod, ExtractFromFunction)):
                            self_pos = self.store_pos_in_caller(pos, idx)
                            if self_pos is None:
                                print(
                                    "\033[34m[warning] cannot find store pos in caller, skip guard check\033[0m",
                                    type(var), var.extract_code_at_start)
                                new_var.need_guard_check = False
                            else:
                                new_var.extract_code_at_start.append(self_pos)
                        else:
                            raise NotImplementedError(pos, type(pos))

        def merge_fx_graph() -> None:
            self.update_subpath(self.root, "")

            def replacement_fn(node: torch.fx.Node) -> torch.fx.Node:
                return replacement_mapping[node]

            def get_tensor_idx(node: torch.fx.Node) -> int:
                var = node.meta["var"]
                if isinstance(var, vs.TensorVar):
                    idx = var.idx
                elif isinstance(var, vs.ScalarVar):
                    idx = id(var.obj)
                else:
                    raise NotImplementedError
                assert isinstance(idx, int) and idx != 0
                return idx

            def get_original_node(node: torch.fx.Node) -> torch.fx.Node:
                idx = get_tensor_idx(node)
                var = node.meta["var"]
                pos = var.extract_code_at_start[0] if len(
                    var.extract_code_at_start) > 0 else None
                if self.objects.contains_by_id(idx):
                    gen_by_caller = self.objects.get_by_id(idx)
                elif pos and isinstance(
                        pos, StoreInAttr
                ) and pos.attr_name == 'data' and self.objects.contains_by_id(
                        pos.self_id):
                    gen_by_caller = self.objects.get_by_id(pos.self_id)
                    assert isinstance(gen_by_caller, vs.TensorVar)
                else:
                    new: list[StorePos] = []
                    for pos in var.extract_code_at_start:
                        new_pos = self.store_pos_in_caller(pos, idx)
                        if new_pos is not None:
                            new.append(new_pos)
                    if len(new) == 0:
                        # the inputs of callee come from generated outputs in caller, should not add to graph as input
                        new_pos = UnknownPosInCaller()
                        new.append(new_pos)
                    new_var = vs.make_var_from_value(
                        var.obj, var.need_guard_check,
                        self.objects.helper_functions, self.fx_graph, new)
                    self.objects.add(new_var, var.obj)
                    gen_by_caller = new_var
                assert isinstance(gen_by_caller, (vs.TensorVar, vs.ScalarVar))
                return gen_by_caller.fx_node

            for node in state.fx_graph.result_graph.nodes:
                if node.op == "placeholder":
                    replacement_mapping[node] = get_original_node(node)
                    continue
                elif node.op == "output":
                    raise ValueError("should not contain output node")
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
                    if param_obj not in self.subparam_paths:
                        self.add_subparam(param_obj)
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
                    if module_obj not in self.submodule_paths:
                        self.add_submodule(module_obj)
                    name_in_caller = self.submodule_paths[module_obj]
                    new_node.target = name_in_caller
                elif node.op == "call_method":
                    pass
                if config.get_config('dynshape'):
                    self.fx_graph.infer_fake_value(new_node)
                replacement_mapping[node] = new_node
                if 'var' in node.meta:
                    var = node.meta['var']
                    assert isinstance(var, vs.Variable)
                    obj_id_need_merge.add(id(var.obj))

        def merge_output() -> None:
            merged_ids: set[int] = set()

            def get_new_store_pos(old: list[StorePos],
                                  idx: int) -> list[StorePos]:
                new: list[StorePos] = []
                for pos in old:
                    new_pos = self.store_pos_in_caller(pos, idx)
                    if new_pos is not None:
                        new.append(new_pos)
                return new

            def get_or_make_var(
                    obj: Any, need_guard_check: bool,
                    fx_graph: Optional[FxGraph],
                    extract_code_at_start: list[StorePos]) -> vs.Variable:
                if id(obj) in merged_ids:
                    new_var = self.objects.get(obj)
                elif self.objects.contains(obj) and self.objects.get(
                        obj, False) is not None:  # value from caller, modified
                    caller_var = self.objects.get(obj, False)
                    if isinstance(caller_var, vs.TensorVar):
                        old_obj = state.objects.get(obj, False)
                        assert isinstance(old_obj, vs.TensorVar)
                        old_node = old_obj.fx_node
                        new_node = replacement_mapping[old_node]
                        new_var = vs.TensorVar.from_tensor_and_node(
                            obj, new_node, need_guard_check,
                            extract_code_at_start)
                    elif is_scalar(obj) and dyn.contains(obj):
                        old_obj = state.objects.get(obj, False)
                        assert isinstance(old_obj, vs.ScalarVar)
                        old_node = old_obj.fx_node
                        assert old_node is not None
                        new_node = replacement_mapping[old_node]
                        new_var = vs.ScalarVar.from_value_and_node(
                            obj, new_node, need_guard_check,
                            extract_code_at_start)
                    else:
                        new_var = vs.make_var_from_value(
                            obj,
                            need_guard_check,
                            vs.HelperFunctions(get_or_make_var,
                                               self.gen_by_caller,
                                               self.mark_cannot_guard),
                            fx_graph,
                            extract_code_at_start,
                        )
                    callee_var = state.objects.get(obj,
                                                   allow_unexist_const=True)
                    for attr_name in callee_var.modified_attrs.keys():
                        attr_obj = getattr(obj, attr_name)
                        if isinstance(attr_obj, (int, float, str)):
                            attr_var = self.objects.get(attr_obj, True)
                        else:
                            if self.objects.contains(attr_obj):
                                attr_var = self.objects.get(attr_obj, False)
                            else:
                                attr_var = state.objects.get(attr_obj, False)
                        assert attr_var is not None
                        new_attr_extract: list[StorePos] = get_new_store_pos(
                            attr_var.extract_code_at_start, id(attr_obj))
                        new_attr_var = get_or_make_var(
                            attr_obj, attr_var.need_guard_check, self.fx_graph,
                            new_attr_extract)
                        new_var.add_modified_attr(attr_name, new_attr_var)
                    self.objects.update_by_id(new_var, id(obj))
                elif self.objects.contains(
                        obj):  # value from caller, unmodified
                    new_var = self.objects.get(obj)
                else:  # value not in caller
                    if isinstance(obj, torch.Tensor) and not isinstance(
                            obj, torch.nn.Parameter):
                        old_obj = state.objects.get(obj, False)
                        assert isinstance(old_obj, vs.TensorVar)
                        old_node = old_obj.fx_node
                        new_node = replacement_mapping[old_node]
                        new_var = vs.TensorVar.from_tensor_and_node(
                            obj, new_node, need_guard_check,
                            extract_code_at_start)
                    elif is_scalar(obj) and dyn.contains(obj):
                        old_obj = state.objects.get(obj, False)
                        assert isinstance(old_obj, vs.ScalarVar)
                        old_node = old_obj.fx_node
                        assert old_node is not None
                        new_node = replacement_mapping[old_node]
                        new_var = vs.ScalarVar.from_value_and_node(
                            obj, new_node, need_guard_check,
                            extract_code_at_start)
                    else:
                        new_var = vs.make_var_from_value(
                            obj, need_guard_check,
                            vs.HelperFunctions(get_or_make_var,
                                               self.gen_by_caller,
                                               self.mark_cannot_guard),
                            fx_graph, extract_code_at_start)
                    self.objects.add(new_var, obj)
                merged_ids.add(id(obj))
                return new_var

            for idx, var in state.objects.get_all_with_id():
                if (var.prev is not None and
                        idx not in ignore_ids) or idx in obj_id_need_merge:
                    oldest = var.get_oldest_var()
                    if len(oldest.extract_code_at_start
                          ) == 0 and idx not in obj_id_need_merge:
                        continue
                    new_extract: list[StorePos] = get_new_store_pos(
                        oldest.extract_code_at_start, idx)
                    get_or_make_var(var.obj, False, self.fx_graph, new_extract)
            if calling_func is None or not is_high_order_func(calling_func):
                for obj in stack_objs:
                    state.fetch_function_parameters(obj)
                    var = state.objects.get(obj, allow_unexist_const=True)
                    new_extract = get_new_store_pos(var.extract_code_at_start,
                                                    id(var.obj))
                    get_or_make_var(obj, False, self.fx_graph, new_extract)

        if len(state.objects.objs_no_id) > 0:
            raise NotImplementedError
        merge_call_guard()
        merge_fx_graph()
        merge_output()
        if isinstance(self.frame_cf_info, IfStmtInfo):
            cond_obj = self.frame_cf_info.cond_obj
            if not self.objects.contains(cond_obj):
                self.fx_graph.result_graph.inserting_before()
                self.objects.add(
                    vs.make_var_from_value(cond_obj, False,
                                           self.objects.helper_functions,
                                           self.fx_graph,
                                           [StoreInLocal('cond')]), cond_obj)
                self.fx_graph.result_graph.inserting_after()
            if not (self.calling_func is not None and
                    hasattr(self.calling_func, '__name__') and
                    self.calling_func.__name__ == 'recover'):
                self.frame_cf_info.mark_end(state.stored_locals,
                                            self.fx_graph.result_graph,
                                            self.objects.get)
        self.object_refs.extend(state.object_refs)
        if calling_func is not None:
            assert len(stack_objs) == 1
            self.callee_returns = stack_objs[0]

    def mark_defer_restart(self,
                           defer_restart_state: DeferRestartState) -> None:
        self.defer_restart = defer_restart_state

    def mark_need_defer_restart(self) -> None:
        assert self.defer_restart is not None
        self.defer_restart.need_restart = True

    def add_inplace_update_obj(self, obj: Any) -> None:
        self.written = True
        self.inplace_update_objs.append(obj)

    def set_partial_var(
            self, partials: dict[int, list[Optional[PartialVar]]]) -> None:
        assert len(self.partial_var) == 0
        self.written = True
        self.partial_var = partials

    def mark_calling_func(self, func: Any) -> None:
        self.calling_func = func

    def unmark_calling_func(self) -> None:
        self.calling_func = None
        self.callee_returns = None

    def mark_cannot_guard(self) -> None:
        self.can_guard = False

    def fetch_function_parameters(self, obj: Any) -> None:
        if not self.objects.contains(obj):
            var = vs.make_var_from_value(obj, False,
                                         self.objects.helper_functions,
                                         self.fx_graph, [])
            self.objects.add_by_id(var, id(obj))


class GuardTracker:
    code: ProcessedCode
    frame_id: int
    frame: FrameType
    state: State
    have_error: bool
    frame_root: torch.nn.Module
    caller: Optional['GuardTracker']
    cf_info: Optional[ControlFlowInfo]
    num_breaks: int
    layout_sensitive: bool

    def __init__(self,
                 frame: FrameType,
                 frame_id: int,
                 caller: Optional['GuardTracker'] = None,
                 read_stack: bool = False,
                 cf_info: Optional[ControlFlowInfo] = None):
        self.code = get_code_map(frame)
        self.frame = frame
        self.frame_id = frame_id
        self.frame_root = get_frame_root(frame_id)
        self.caller = caller
        if self.caller is not None and self.caller.state.calling_func == if_stmt:
            assert cf_info is None
            cond_obj = frame.f_locals['cond']
            cond_as_bool = bool(cond_obj)
            if_true = frame.f_locals['if_true']
            if_false = frame.f_locals['if_false']
            cf_info = IfStmtInfo(0,
                                 len(self.code.original_insts) - 1, if_true,
                                 if_false, cond_obj, cond_as_bool,
                                 self.frame_root)
            f_locals = self.frame.f_locals
            if_other_id = self.frame.f_code.co_varnames.index('if_other_branch')
            if_run_id = self.frame.f_code.co_varnames.index('if_run_branch')
            # TODO: check the type of cond
            if cond_as_bool:
                set_local(self.frame, if_other_id, f_locals['if_false'])
                set_local(self.frame, if_run_id, f_locals['if_true'])
            else:
                set_local(self.frame, if_other_id, f_locals['if_true'])
                set_local(self.frame, if_run_id, f_locals['if_false'])
        self.cf_info = cf_info
        self.init_state(
            read_stack=read_stack, frame_cf_info=cf_info
        )  # stack pointer is not initialized at the creation of a stack frame
        self.num_breaks = 0
        self.layout_sensitive = False

    def init_state(self,
                   read_stack: bool = True,
                   frame_cf_info: Optional[ControlFlowInfo] = None) -> None:
        if hasattr(self, "state"):
            self.state.written = False
        self.state = State.from_frame(self.frame, self.frame_id, read_stack,
                                      self.frame_root, self.gen_by_caller,
                                      frame_cf_info)
        self.have_error = False

    def record(
            self, frame: FrameType, frame_id: int
    ) -> None:  # pass frame and frame_id only for assertion
        assert frame_id == self.frame_id
        assert frame == self.frame, (frame, self.frame)
        self.process_last_inst()

        pc, inst = self.code.get_orig_inst(self.frame.f_lasti)
        if self.cf_info is not None and pc == self.cf_info.end_pc:
            self.restart("reach end of nested tracker", restart_caller=False)
            return
        if inst is None:
            self.restart(
                f"running injected code (f_lasti={self.frame.f_lasti})",
                restart_caller=False)
            if self.code.get_inst(self.frame.f_lasti).opname == 'RETURN_VALUE':
                if trackers[-1] == self:
                    if self.layout_sensitive == True:
                        if self.caller is not None:
                            self.caller.layout_sensitive = True
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
                self.init_state(frame_cf_info=self.cf_info)
            except Exception as e:
                self.restart(f"Exception during init: {e}")
                print(traceback.format_exc())
                # raise e
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
                self.state.guarded_pcs.append(self.frame.f_lasti // 2)
                getattr(self, inst.opname)(inst)
                # NOTE: DO NOT write any function call after this line
                # because frame evaluation function may be set during processing the opcode
            except Exception as e:
                print(traceback.format_exc())
                raise e
                self.restart(f"Exception during processing {inst.opname}: {e}")
            if not self.have_error and self.state.defer_restart is None:
                self.state.is_empty = False
                self.state.written = False
        else:
            raise ValueError(f"unknown opcode {inst.opname}")
            self.restart(f"unknown opcode {inst.opname}")

    def commit_loop_subgraph(self) -> None:
        key = new_random_key()
        guard_codegen = GuardFnCodegen(key=key)
        if self.layout_sensitive == True:
            guard_codegen.layout_sensitive = True
        for var in self.state.objects.get_all():
            while var.prev is not None:
                var = var.prev
            var.make_guard(guard_codegen)
        guard_code = guard_codegen.get_code()
        out: Dict[str, Any] = dict()
        exec(guard_code, self.frame.f_globals, out)
        guard_fn = out["___make_guard_fn"](*guard_codegen.objs.values())
        frame_locals = self.frame.f_locals
        stack_locals = {
            f"__stack__{i}": get_value_stack_from_top(self.frame, i)
            for i in range(get_value_stack_size(self.frame))
        }
        fill_locals = stack_locals | frame_locals
        cf_info = self.cf_info
        if guard_fn(fill_locals):
            print("guard fn success, can generate loop")
            fx_graph = self.state.fx_graph
            pos2input: dict[str, tuple[StorePos, torch.fx.Node]] = {}
            for fx_node in fx_graph.result_graph.nodes:
                if fx_node.op == "placeholder":
                    var = fx_node.meta["var"]
                    assert len(var.extract_code_at_start) > 0
                    for pos in var.extract_code_at_start:
                        pos2input[str(pos)] = (pos, fx_node)

            pos2output: dict[str, tuple[StorePos, torch.fx.Node]] = {}

            live_objs = self.get_live_objs(self.state.guarded_pcs[-2])

            for live_name, obj in live_objs:
                if not isinstance(obj, torch.Tensor) and not is_scalar(obj):
                    raise NotImplementedError
                fx_node = self.state.objects.get(obj).as_fx_node()
                assert isinstance(fx_node, torch.fx.Node)
                if live_name in self.frame.f_locals:
                    pos = StoreInLocal(live_name)
                    pos2output[str(pos)] = (pos, fx_node)
                elif live_name in self.frame.f_globals:
                    pos = StoreInGlobal(live_name)
                    pos2output[str(pos)] = (pos, fx_node)
                else:
                    raise NotImplementedError
            for var in self.state.objects.get_all():
                if var.prev is not None:
                    if isinstance(var, vs.RangeIterVar):
                        continue
                    oldest = var.get_oldest_var()
                    for pos in oldest.extract_code_at_start:
                        fx_node = var.as_fx_node()
                        assert isinstance(fx_node, torch.fx.Node)
                        pos2output[str(pos)] = (pos, fx_node)
            input_only_pos: list[tuple[str, StorePos]] = []
            joint_pos: list[tuple[str, StorePos]] = []
            output_only_pos: list[tuple[str, StorePos]] = []
            for pos_str, (pos, _) in pos2input.items():
                if pos_str in pos2output:
                    joint_pos.append((pos_str, pos))
                else:
                    input_only_pos.append((pos_str, pos))
            for pos_str, (pos, _) in pos2output.items():
                if pos_str not in pos2input:
                    output_only_pos.append((pos_str, pos))

            if len(output_only_pos) > 0:
                raise NotImplementedError

            input_only_pos.sort()
            joint_pos.sort()
            output_only_pos.sort()

            replacement_mapping: dict[torch.fx.Node, torch.fx.Node] = {}

            def replacement_fn(node: torch.fx.Node) -> torch.fx.Node:
                return replacement_mapping[node]

            inner_fx_graph = torch.fx.Graph()
            for pos_str, _ in input_only_pos:
                _, old_node = pos2input[pos_str]
                new_node = inner_fx_graph.placeholder(old_node.name,
                                                      old_node.type)
                replacement_mapping[old_node] = new_node
            for pos_str, _ in joint_pos:
                _, old_node = pos2input[pos_str]
                new_node = inner_fx_graph.placeholder(old_node.name,
                                                      old_node.type)
                replacement_mapping[old_node] = new_node
            for node in fx_graph.result_graph.nodes:
                if node.op != "placeholder":
                    new_node = inner_fx_graph.node_copy(node, replacement_fn)
                    replacement_mapping[node] = new_node
            output_nodes = []
            for pos_str, _ in joint_pos:
                _, old_node = pos2output[pos_str]
                output_nodes.append(replacement_fn(old_node))
            inner_fx_graph.output(tuple(output_nodes))

            cf_info = self.cf_info
            assert isinstance(cf_info, ForLoopInfo)
            cf_info.inner_graph = inner_fx_graph
            cf_info.pos_map = LoopPosMap(input_only_pos, joint_pos,
                                         output_only_pos)
            print("new fx graph", inner_fx_graph)
            print("posmap", cf_info.pos_map)
        else:
            raise NotImplementedError("TODO")

    def rewrite_loop_graph(self) -> None:
        fx_graph = self.state.fx_graph
        input_nodes: dict[str, torch.fx.Node] = {}
        loop_info = self.cf_info
        assert isinstance(loop_info, ForLoopInfo)
        pos_map = loop_info.pos_map
        assert pos_map is not None
        body_graph = loop_info.inner_graph
        assert body_graph is not None
        body_graph_module = torch.fx.GraphModule(
            self.frame_root,
            body_graph,
        )
        num_input_only_pos = len(pos_map.input_only_pos)
        for _, pos in pos_map.input_only_pos:
            if isinstance(pos, IterValue):
                num_input_only_pos -= 1
        loop_module = LoopModule(body_graph_module, num_input_only_pos,
                                 loop_info.num_iter)
        loop_module_name = new_name("__loop_module__")
        self.frame_root.add_module(loop_module_name, loop_module)
        self.state.submodule_paths[loop_module] = loop_module_name
        iter_value_str = str(IterValue())
        for node in fx_graph.result_graph.nodes:
            if node.op == "placeholder":
                var = node.meta["var"]
                assert len(var.extract_code_at_start) > 0
                for pos in var.extract_code_at_start:
                    if isinstance(pos, IterValue):
                        continue
                    input_nodes[str(pos)] = node
            elif node.op == "output":
                fx_graph.result_graph.inserting_before(node)
        input_args = [
            input_nodes[p]
            for p, _ in itertools.chain(pos_map.input_only_pos,
                                        pos_map.joint_pos)
            if p != iter_value_str
        ]
        output_args = []
        output_vars = []
        for _, pos in itertools.chain(pos_map.joint_pos,
                                      pos_map.output_only_pos):
            obj = pos.get_value_from_frame(self.frame)
            var = self.state.objects.get(obj)
            assert isinstance(var, (vs.TensorVar, vs.ScalarVar))
            output_args.append(var.as_fx_node())
            output_vars.append(var)
        new_nodes = []
        loop_node = fx_graph.result_graph.call_module(loop_module_name,
                                                      tuple(input_args))
        new_nodes.append(loop_node)
        node_map: dict[torch.fx.Node, torch.fx.Node] = {}
        for i, (old_node, var) in enumerate(zip(output_args, output_vars)):
            new_node = fx_graph.result_graph.call_function(
                operator.getitem, (loop_node, i))
            new_node.meta["var"] = var
            var.fx_node = new_node
            new_nodes.append(new_node)
            node_map[old_node] = new_node
        all_nodes = list(fx_graph.result_graph.nodes)
        graph_outputs: list[torch.fx.Node] = []
        for node in reversed(all_nodes):
            if node.op == 'output':
                for old_node, new_node in node_map.items():
                    node.replace_input_with(old_node, new_node)
            elif node.op == 'placeholder':
                if len(node.users) == 0:
                    fx_graph.result_graph.erase_node(node)
            else:
                if node not in new_nodes:
                    fx_graph.result_graph.erase_node(node)

    def commit(self) -> None:
        assert not self.state.written
        if self.state.is_empty:
            return
        assert self.state.start_pc >= 0
        if self.state.defer_restart is not None:
            lasti = self.state.defer_restart.guard_lasti
        else:
            lasti = self.frame.f_lasti
        end_pc = self.code.get_orig_pc(lasti)
        if end_pc == -1:
            end_pc = self.code.get_next_orig_pc(lasti)
        print("commiting", self.frame_id, self.state.start_pc, end_pc,
              self.code.original_insts[end_pc], lasti)
        # TODO: can be optimized by only reproduce the modified variables
        if self.state.defer_restart is not None:
            stack_objs = self.state.defer_restart.stack_objs
            assert stack_objs is not None
        else:
            stack_objs = get_all_objects_in_stack(self.frame)

        if self.state.start_pc == 0 and self.code.original_insts[
                end_pc].opname == "RETURN_VALUE" and self.caller is not None:
            print("callee is full graph, merge to caller")
            assert len(stack_objs) == 1
            caller = self.caller
            assert caller is not None
            caller.state.merge_call(self.state,
                                    [get_value_stack_from_top(self.frame, 0)])
        elif self.cf_info is not None and self.num_breaks == 1 and self.cf_info.end_pc == end_pc:
            print("reach end of nested tracker, merge to caller")
            self.rewrite_loop_graph()
            stack_objs = get_all_objects_in_stack(self.frame)
            nest_caller = self.caller
            assert nest_caller is not None
            nest_caller.state.merge_call(self.state, stack_objs)
        else:
            if self.state.can_guard:
                key = new_random_key()
                guard_codegen = GuardFnCodegen(key=key)
                if self.layout_sensitive == True:
                    guard_codegen.layout_sensitive = True
                for var in self.state.objects.get_all():
                    while var.prev is not None:
                        var = var.prev
                    var.make_guard(guard_codegen)
                if config.get_config('dynshape'):
                    self.state.fx_graph.make_shape_env_guard(guard_codegen)
                guard_code = guard_codegen.get_code()
                graph_codegen = GraphFnCodegen(key=key)
                for node in self.state.fx_graph.result_graph.nodes:
                    if node.op == "placeholder":
                        var = node.meta["var"]
                        if isinstance(var, vs.TensorVar):
                            if len(var.extract_code_at_start) == 0:
                                var = var.get_oldest_var()
                            pos = var.extract_code_at_start[0]
                            # input comes from freevar?
                            if isinstance(pos, StoreInAttr) and isinstance(
                                    pos.self_pos, StoreInFreeVar):
                                pos.self_pos.add_name_to_fn(graph_codegen)
                            graph_codegen.add_graph_input(
                                var.extract_code_at_start[0])
                        elif isinstance(var, vs.ScalarVar):
                            graph_codegen.add_graph_input(
                                var.extract_code_at_start[0], to_tensor=True)
                        else:
                            raise ValueError("unknown var type", var)
                current_inst = self.code.get_inst(lasti)
                # livevars_analysis should return the same result when passing self.code.guard_insts
                # and self.code.original_insts, but as current_inst may not be in original_insts,
                # we pass guard_insts here
                if self.state.defer_restart is not None:
                    live_vars = self.state.defer_restart.live_vars
                else:
                    live_vars = self.get_live_objs()

                for i, (live_name, live_obj) in enumerate(live_vars):
                    var = self.state.objects.get(live_obj,
                                                 allow_unexist_const=True)
                    var.make_output(f"__live_{i}", StoreInLocal(live_name),
                                    graph_codegen, True, id(live_obj))

                for i, value in enumerate(stack_objs):
                    var = self.state.objects.get(value,
                                                 allow_unexist_const=True)
                    var.make_output(f"__stack__{i}", StoreInStack(i),
                                    graph_codegen, True, id(value))

                for idx, var in self.state.objects.get_all_with_id():
                    if var.prev is not None and idx not in graph_codegen.id2name:
                        oldest_var = var.get_oldest_var()
                        if len(oldest_var.extract_code_at_start) == 0:
                            continue
                        var.make_output(f"__tmp_{idx}",
                                        oldest_var.extract_code_at_start[0],
                                        graph_codegen, False, idx)

                self.state.fx_graph.set_output_nodes(
                    graph_codegen.get_graph_outputs())
                print("graph input", [
                    (name, x) for x, name in self.state.fx_graph.example_inputs
                ])
                print("graph", self.state.fx_graph.result_graph)
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
                guard_fn = out["___make_guard_fn"](*guard_codegen.objs.values())
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
            obj = get_value_stack_from_top(self.frame, i)
            self.state.object_refs.append(obj)
            if isinstance(obj, super):
                super_var = vs.make_var_from_value(
                    obj, False, self.state.objects.helper_functions,
                    self.state.fx_graph, [])
                self.state.objects.add_by_id(super_var, id(obj))
        self.state.num_new_refs = 0
        for i, obj in enumerate(self.state.inplace_update_objs):
            assert not isinstance(obj, torch.Tensor)
            new_var = vs.make_var_from_value(
                obj, False, self.state.objects.helper_functions,
                self.state.fx_graph, [])
            self.state.objects.update_by_id(new_var, id(obj))

        if -1 in self.state.partial_var:
            partials = self.state.partial_var[-1]
        elif self.frame.f_lasti // 2 in self.state.partial_var:
            partials = self.state.partial_var[self.frame.f_lasti // 2]
        else:
            partials = []
        for i, partial in enumerate(partials):
            if partial is None:
                continue
            value = get_value_stack_from_top(self.frame, i)
            node = partial.node
            if partial.named_class:
                for j in self.state.named_funcs:
                    if j.cls_name == value.__name__:
                        j.obj_class = value
                        break
                continue
            if partial.named_func:
                for j in self.state.named_funcs:
                    if isinstance(value, j.obj_class):
                        j.obj = value
                        self.state.objects.add(j, value)
                        break
                continue
            if partial.force_new_value and self.state.objects.contains(value):
                if isinstance(value, (float, int)):
                    new_value = -(-value)
                    assert id(value) != id(new_value)
                    set_value_stack_from_top(self.frame, i, new_value)
                    value = new_value
                elif isinstance(value, (bool,)):
                    raise NotImplementedError
                elif isinstance(value, torch.Tensor):
                    new_value = torch.clone(value)
                    assert id(value) != id(new_value)
                    set_value_stack_from_top(self.frame, i, new_value)
                    value = new_value
                else:
                    raise ValueError("duplicate id", value)
            default_make_var_fn: MAKE_VAR_FN_TYPE = vs.make_var_from_value
            partial_make_var_fn: Optional[
                MAKE_VAR_FN_TYPE] = partial.make_var_fn
            make_var_fn: MAKE_VAR_FN_TYPE = partial_make_var_fn if partial_make_var_fn is not None else default_make_var_fn
            if isinstance(value, bool) and config.get_config(
                    "dynshape") and node is not None:
                fake = node.meta["fake"]
                if isinstance(fake, torch.SymBool):
                    fake_bool = fake.node.expr
                    import sympy
                    if fake_bool is sympy.true or fake_bool is sympy.false:  # not a dynamic value
                        node = None
            if isinstance(value, torch.Tensor):
                if isinstance(value, torch.nn.Parameter):
                    var = make_var_fn(value, partial.need_guard_check,
                                      self.state.objects.helper_functions,
                                      self.state.fx_graph,
                                      partial.extract_code_at_start)
                elif node is None:
                    if self.state.objects.contains(value):
                        var = self.state.objects.get(value)
                    else:
                        var = vs.TensorVar.from_value(
                            value, partial.need_guard_check,
                            self.state.objects.helper_functions,
                            self.state.fx_graph, partial.extract_code_at_start)
                        # raise ValueError("Unknown node for tensor object")
                else:
                    var = vs.TensorVar.from_tensor_and_node(
                        value, node, partial.need_guard_check,
                        partial.extract_code_at_start)
            elif is_scalar(value) and node is not None:
                dyn.mark_dynamic(value, dyn.ScalarWithUnknownValue())
                var = vs.ScalarVar.from_value_and_node(
                    value,
                    partial.node,
                    partial.need_guard_check,
                    partial.extract_code_at_start,
                )
            elif node is not None:

                def make_sub_var(value: Any, fx_node: torch.fx.Node) -> None:
                    if isinstance(value, torch.Tensor):
                        # if self.state.objects.contains(value):
                        #     raise NotImplementedError
                        new_var = vs.TensorVar.from_tensor_and_node(
                            value, fx_node, partial.need_guard_check,
                            partial.extract_code_at_start)
                        self.state.objects.add(new_var, value)
                    elif is_scalar(value):
                        assert isinstance(value,
                                          (int, float)), "not implemented"
                        if self.state.objects.contains(value):
                            if dyn.contains(value):
                                return
                            # raise NotImplementedError
                        new_scalar_var = vs.ScalarVar.from_value_and_node(
                            value, fx_node, partial.need_guard_check,
                            partial.extract_code_at_start)
                        self.state.objects.add(new_scalar_var, value)
                        dyn.mark_dynamic(value, dyn.ScalarWithUnknownValue())
                    elif value is None:
                        new_none_var = vs.NoneVar.from_value(
                            None, False, self.state.objects.helper_functions,
                            None, partial.extract_code_at_start)
                        self.state.objects.add(new_none_var, None)
                    elif isinstance(value, (tuple, list)):
                        for i, sub_value in enumerate(value):
                            sub_node = self.state.fx_graph.create_node(
                                "call_function", operator.getitem, (fx_node, i),
                                {})
                            make_sub_var(sub_value, sub_node)
                    else:
                        print("tuple inner unknown node", value, type(value))
                        raise NotImplementedError(type(value))

                if isinstance(value, (tuple, list)):
                    make_sub_var(value, node)
                elif inspect.isclass(type(value)):
                    pass
                else:
                    print("partial node with unknown value", value, type(value))
                    raise NotImplementedError
                var = make_var_fn(value, partial.need_guard_check,
                                  self.state.objects.helper_functions,
                                  self.state.fx_graph,
                                  partial.extract_code_at_start)
            else:

                var = make_var_fn(value, partial.need_guard_check,
                                  self.state.objects.helper_functions,
                                  self.state.fx_graph,
                                  partial.extract_code_at_start)
            if partial.inplace_ref is None:
                self.state.objects.add(var, value)
            else:
                self.state.objects.update_by_id(var, id(value))

        if self.state.calling_func is not None:
            stack_top = get_value_stack_from_top(self.frame, 0)
            if self.state.callee_returns is not None and id(
                    self.state.callee_returns) != id(
                        stack_top
                    ):  # a typical case is test_call_udf/test_call_run_udf
                if isinstance(self.state.callee_returns,
                              torch.Tensor) and isinstance(
                                  stack_top, torch.Tensor):
                    if self.state.objects.contains(stack_top):
                        raise NotImplementedError
                    returns_var = self.state.objects.get(
                        self.state.callee_returns)
                    new_node = self.state.fx_graph.create_node(
                        "call_function", torch.clone,
                        (returns_var.as_fx_node(),), {})
                    stack_top_var = vs.TensorVar.from_tensor_and_node(
                        stack_top, new_node, False, [])
                    self.state.add_object(stack_top_var, stack_top)
                elif is_high_order_func(self.state.calling_func):  # zip / map
                    pass
                elif isinstance(self.state.callee_returns,
                                tuple) and isinstance(stack_top, tuple):
                    flag = all(
                        torch.equal(i, j) if isinstance(i, torch.Tensor
                                                       ) else i == j
                        for i, j in zip(self.state.callee_returns, stack_top))
                    assert flag
                    # handle sub values
                    for callee, stack in zip(self.state.callee_returns,
                                             stack_top):
                        sub_var = self.state.objects.get(callee)
                        if isinstance(callee, torch.Tensor):
                            new_node = self.state.fx_graph.create_node(
                                "call_function", torch.clone,
                                (sub_var.as_fx_node(),), {})
                            stack_sub_var: Variable = vs.TensorVar.from_tensor_and_node(
                                stack, new_node, False, [])
                        else:
                            stack_sub_var = vs.make_var_from_value(
                                stack, False,
                                self.state.objects.helper_functions,
                                self.state.fx_graph,
                                sub_var.extract_code_at_start)
                        self.state.add_object(stack_sub_var, stack)
                    # handle tuple value
                    returns_var = self.state.objects.get(
                        self.state.callee_returns)
                    top_var = vs.tuple_.TupleVar.from_value(
                        stack_top, returns_var.need_guard_check,
                        self.state.objects.helper_functions,
                        self.state.fx_graph, returns_var.extract_code_at_start)
                    self.state.add_object(top_var, stack_top)
                else:
                    raise NotImplementedError

        self.state.inplace_update_objs.clear()
        self.state.partial_var.clear()
        self.state.written = False
        self.state.unmark_calling_func()
        # print('process last instruction done')
        if self.state.defer_restart is not None and self.state.defer_restart.need_restart:
            self.restart("defered restart " + self.state.defer_restart.reason)
        self.state.defer_restart = None

    def restart(self, restart_reason: str, restart_caller: bool = True) -> None:
        print(f"restart: {restart_reason}")
        self.have_error = True
        self.num_breaks += 1
        self.commit()
        if self.caller is not None and restart_caller:
            self.caller.state.mark_need_defer_restart()
        if self.cf_info is not None and self.cf_info.end_pc == self.code.get_orig_pc(
                self.frame.f_lasti):
            self.state.is_empty = True
            pop_tracker(self.frame_id)

    @classmethod
    def has_tensor_arg(cls, args: List[Any], kwargs: Dict[str, Any]) -> bool:
        return any(
            isinstance(i, torch.Tensor) or (is_scalar(i) and dyn.contains(i))
            for i in itertools.chain(args, kwargs.values()))

    @classmethod
    def all_scalar_arg(cls, args: List[Any], kwargs: Dict[str, Any]) -> bool:
        return all(is_scalar(i) for i in itertools.chain(args, kwargs.values()))

    @classmethod
    def all_static_arg(cls, args: List[Any], kwargs: Dict[str, Any]) -> bool:
        return all(
            not dyn.contains(i) for i in itertools.chain(args, kwargs.values()))

    @classmethod
    def has_ndarray_arg(cls, args: List[Any], kwargs: Dict[str, Any]) -> bool:
        import numpy
        return any(
            isinstance(i, numpy.ndarray)
            for i in itertools.chain(args, kwargs.values()))

    @classmethod
    def has_arg_of_type(
            cls, args: List[Any], kwargs: Dict[str, Any],
            arg_type: Union[type[Any], tuple[type[Any], ...]]) -> bool:
        return any(
            isinstance(i, arg_type)
            for i in itertools.chain(args, kwargs.values()))

    def has_unknown_arg(self, args: List[Any], kwargs: Dict[str, Any]) -> bool:
        return any(
            isinstance(self.state.objects.get_or_none(i), vs.AnyVar)
            for i in itertools.chain(args, kwargs.values()))

    def has_no_arg(self, args: List[Any], kwargs: Dict[str, Any]) -> bool:
        return not (len(args) or len(kwargs))

    def is_genexpr_func(self, func: Callable[..., Any]) -> bool:
        return (hasattr(func, '__name__') and func.__name__ == '<genexpr>')

    def is_builtin_func(self, func: Callable[..., Any]) -> bool:
        return func in (dict, tuple, set, list, hasattr, slice, range, len,
                        type, all, str.join, reversed, zip, iter, id, next,
                        collections.OrderedDict, str.format, any, str,
                        str.split)

    def is_numpy_constant_func(self, func: Callable[..., Any]) -> bool:
        print(dir(func))
        if (hasattr(func, '__module__') and 'numpy' in func.__module__ and
                'random' not in func.__module__):
            return True
        if type(func) == np.ufunc:
            return True
        return False

    def get_live_objs(self, pc: int = -1) -> list[tuple[str, Any]]:
        if pc == -1:
            pc = self.frame.f_lasti // 2
        live_names = livevars_analysis(self.code.guard_insts,
                                       self.code.get_inst(pc * 2))
        live_names = live_names.intersection(self.state.stored_locals)
        return [(name, self.frame.f_locals[name]) for name in live_names]

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
        if func == operator.is_ and args[1] is None:  # is_none check
            return
        if func == enumerate:
            assert len(args) == 1
            assert len(kwargs) == 0
            var = self.state.objects.get_or_none(args[0])
            assert var is not None
            new_store_pos: list[StorePos] = [
                ExtractFromFunction([pos], [id(args[0])], func.__name__, func)
                for pos in var.extract_code_at_start
            ]
            self.state.set_partial_var({
                -1: [
                    PartialVar(node=None,
                               need_guard_check=False,
                               extract_code_at_start=new_store_pos)
                ]
            })
        if func == super:
            obj = self.state.initial_args[0]
            name = self.frame.f_code.co_varnames[0]
            if name not in self.state.stored_locals:
                if not self.state.objects.contains(obj):
                    pos = StoreInLocal(name)
                    var = vs.make_var_from_value(
                        obj, True, self.state.objects.helper_functions,
                        self.state.fx_graph, [pos])
                    self.state.add_object(var, obj)
            return
        if hasattr(func,
                   '__name__') and func.__name__ == 'format' and isinstance(
                       func, type(str.format)):
            for arg in args:
                if torch.is_tensor(arg) or dyn.contains(arg):
                    raise ValueError("format can't have dynamic args")
        if hasattr(func, '__name__') and (func.__name__ == 'is_contiguous' or
                                          func.__name__ == 'stride'):
            self.layout_sensitive = True
        if hasattr(func, '__name__') and func.__name__ == '__init__':
            return
        # a series of classes and functions defined by warnings
        if get_root_module(func) in ('_warnings', 'warnings'):
            return
        if get_root_module(func) == 'random':
            for arg in args:
                if torch.is_tensor(arg) or dyn.contains(arg):
                    raise ValueError("random func can't have dynamic args")
            if func.__name__ not in {
                    'random', 'randint', 'randrange', 'uniform'
            }:
                raise ValueError("Not implement random func")

            name = new_name('random')
            fx_node = self.state.fx_graph.create_input(torch.tensor([0]), name,
                                                       (), {}, name)
            self.state.set_partial_var({
                -1: [
                    PartialVar(
                        node=fx_node,
                        need_guard_check=False,
                        extract_code_at_start=[
                            ExtractFromFunction(
                                [StoreConstant(arg, id(arg)) for arg in args],
                                [id(arg) for arg in args], func.__name__, func,
                                True)
                        ])
                ]
            })
            return
        is_high_order_udf = is_high_order_func_with_udf(func, args, kwargs)
        if is_user_defined_func(func) or isinstance(
                func, nn.Sequential) or is_high_order_udf:
            if isinstance(self.cf_info, IfStmtInfo):
                if hasattr(func, '__name__') and func.__name__ == 'recover':
                    self.cf_info.recover()
                else:
                    self.cf_info.mark_start()
            if inspect.isclass(func) and not is_high_order_udf:
                for i in self.state.named_funcs:
                    if func == i.obj_class:
                        i.attr_value = args
                        self.state.set_partial_var({
                            -1: [
                                PartialVar(node=None,
                                           need_guard_check=False,
                                           extract_code_at_start=[],
                                           named_func=True)
                            ]
                        })
                        return
                class_define_new = get_method_defined_class(func, '__new__')
                if class_define_new not in (object, torch._C._FunctionBase,
                                            dict):
                    raise NotImplementedError("user defined __new__",
                                              class_define_new)
                class_define_init = get_method_defined_class(func, '__init__')
                if class_define_init in (
                        torch.autograd.function.InplaceFunction,
                        torch.autograd.function.Function, object, dict,
                        torch.nn.modules.module.Module):
                    self.state.set_partial_var({
                        -1: [
                            PartialVar(node=None,
                                       need_guard_check=False,
                                       extract_code_at_start=[
                                           ExtractFromFunction([], [],
                                                               func.__name__,
                                                               func, True)
                                       ])
                        ]
                    })
                    return
            if hasattr(func, "__func__"
                      ) and func.__func__ == torch.nn.Sequential.parameters:
                method_var = self.state.objects.get(func)
                method_pos = method_var.extract_code_at_start[0]
                assert isinstance(method_pos, StoreInAttr)
                self.state.set_partial_var({
                    -1: [
                        PartialVar(node=None,
                                   need_guard_check=False,
                                   extract_code_at_start=[
                                       ExtractFromMethod(
                                           method_pos.self_pos,
                                           id(method_var.obj),
                                           method_pos.attr_name)
                                   ])
                    ]
                })
                return
            print("run into user defined function", func)
            stack_objs = get_all_objects_in_stack(self.frame)
            self.state.mark_calling_func(func)
            self.state.mark_defer_restart(
                DeferRestartState(stack_objs, self.get_live_objs(),
                                  self.frame.f_lasti, f"call_function", False))
            from .tracer import get_process_frame
            preprocess_frame, post_process_frame = get_process_frame(func, True)
            prior = set_eval_frame((preprocess_frame, post_process_frame))
            assert prior is None
            # assert self.state.written == False
            return
        if func in fx_graph_inplace_functions or (hasattr(
                func, '__name__') and func.__name__ in torch_inplace_funcs):
            if len(args) == 0:
                raise NotImplementedError
            if not self.state.objects.contains(args[0]):
                self.state.add_object(
                    vs.make_var_from_value(args[0], False,
                                           self.state.objects.helper_functions,
                                           self.state.fx_graph, []), args[0])
            inplace_ref = args[0]
        else:
            inplace_ref = None

        def set_if_inplace_return() -> None:
            if inplace_ref is not None:
                self.state.set_partial_var({
                    -1: [
                        PartialVar(node=None,
                                   need_guard_check=False,
                                   extract_code_at_start=[],
                                   inplace_ref=inplace_ref)
                    ]
                })

        pc, inst = self.code.get_orig_inst(self.frame.f_lasti)
        if is_graph_func(func):
            has_ndarray_flag = self.has_ndarray_arg(args, kwargs)
        else:
            has_ndarray_flag = False
        if len(args) == 1 and isinstance(args[0],
                                         (tuple, list)) and func != len:
            has_tensor_flag = self.has_tensor_arg(list(args[0]), kwargs)
        else:
            has_tensor_flag = self.has_tensor_arg(args, kwargs)
        if len(args) > 0 and isinstance(
                args[0], (tuple, list)) and func == operator.getitem:
            has_tensor_flag = self.has_tensor_arg(list(args[0]), kwargs)
        if get_root_module(func) == 'torch' or (
                has_tensor_flag and
            (is_graph_func(func) or is_math_func(func) or
             func in (float, int, min, max, len, list, abs, sum))):
            if hasattr(func, "__name__") and (
                    func.__name__ in
                ("named_children", "_are_functorch_transforms_active", "finfo",
                 "dim", "save_for_backward", "_get_tracing_state", "len") or
                (func.__name__ == "type" and inst is not None and
                 inst.argval == 0) or (func.__name__ in ("size",) and
                                       not config.get_config("dynshape"))):
                self.state.set_partial_var({
                    -1: [
                        PartialVar(node=None,
                                   need_guard_check=False,
                                   extract_code_at_start=[])
                    ]
                })
                return
            if hasattr(func, "__name__") and func.__name__ in (
                    "flatten_parameters", "numel", "children",
                    "named_parameters", "_weights_have_changed",
                    "check_forward_args", "permute_hidden", "_check_input_dim",
                    "parameters", "_has_torch_function_unary", "_is_tracing",
                    "is_tracing", "is_scripting", "get_autocast_gpu_dtype",
                    "is_autocast_enabled", "ndimension", "get_enum",
                    "is_tensor", "is_complex", "is_contiguous", "stride",
                    "get_device"):
                return
            if hasattr(func, "__module__"
                      ) and func.__module__ == 'torch.autograd.profiler':
                return
            elif hasattr(func, "__self__") and isinstance(
                    func.__self__, torch.autograd.profiler.record_function):
                return
            print("record function in graph", func)
            self.state.record_function(
                func,
                args,
                kwargs,
                inplace_ref=inplace_ref,
                force_new_value=(func in (float, int, min, max) or
                                 (hasattr(func, '__name__') and
                                  func.__name__ == 'contiguous')))
            return
        elif self.all_scalar_arg(args, kwargs) and self.all_static_arg(
                args, kwargs):
            if func == range:
                self.state.set_partial_var({
                    -1: [
                        PartialVar(node=None,
                                   need_guard_check=False,
                                   extract_code_at_start=[])
                    ]
                })
            return
        elif get_root_module(func) == 'numpy' or has_ndarray_flag:
            print("record numpy function in graph", func)
            # self.state.record_function(func,
            #                            args,
            #                            kwargs,
            #                            inplace_ref=inplace_ref,
            #                            force_new_value=False)
            self.state.set_partial_var({
                -1: [
                    PartialVar(node=None,
                               need_guard_check=False,
                               extract_code_at_start=[])
                ]
            })
            return
        elif self.has_arg_of_type(args, kwargs, tuple):
            return
        elif self.has_arg_of_type(
                args, kwargs,
            (set, list, dict, collections.OrderedDict,
             MappingProxyType)) and get_root_module(func) != 'torch':
            if hasattr(func, "__name__") and func.__name__ == 'namedtuple':
                assert len(args) == 2
                cls_by_define = ClsByNamedTupleVar(
                    args[0], args[1], False, None, [],
                    self.state.objects.helper_functions)
                self.state.named_funcs.append(cls_by_define)
                self.state.set_partial_var({
                    -1: [
                        PartialVar(node=None,
                                   need_guard_check=False,
                                   extract_code_at_start=[],
                                   named_class=True)
                    ]
                })
            set_if_inplace_return()
            if len(args) > 0 and isinstance(
                    args, list) and func in (list.append, list.extend,
                                             list.clear, list.pop, list.remove,
                                             list.reverse, list.sort):
                self.state.add_inplace_update_obj(args[0])
            return
        elif self.has_arg_of_type(args, kwargs, np.generic):
            return
        elif self.is_genexpr_func(func):
            return
        elif self.is_builtin_func(func):
            self.state.set_partial_var({
                -1: [
                    PartialVar(node=None,
                               need_guard_check=False,
                               extract_code_at_start=[])
                ]
            })
            return
        elif func in (super, map, filter, enumerate):
            # TODO: add map and set correct partial var
            return
        elif is_graph_func(func):
            return
        elif len(args) > 0 and isinstance(args[0], torch.nn.ModuleList):
            return
        elif self.is_numpy_constant_func(func):
            return
        elif self.has_unknown_arg(args, kwargs):
            print(
                f"func is {func}, {is_user_defined_func(func)}, args: {args}, kwargs:{kwargs}"
            )
            raise NotImplementedError
        elif func == getattr:
            if get_method_defined_class(type(args[0]), '__getattr__') in (
                    torch.nn.Module, object, None) and get_method_defined_class(
                        type(args[0]),
                        '__getattribute__') in (torch.nn.Module, object,
                                                ModuleType):
                arg_obj = self.state.objects.get(args[0])

                self.state.set_partial_var({
                    -1: [
                        PartialVar(node=None,
                                   need_guard_check=arg_obj.need_guard_check,
                                   extract_code_at_start=[
                                       StoreInAttr(p, id(args[0]), args[1])
                                       for p in arg_obj.extract_code_at_start
                                   ])
                    ]
                })
                return
            return
        elif func == setattr:
            return
        elif func == isinstance:
            return
        elif func == inspect.signature:
            self.state.set_partial_var({
                -1: [
                    PartialVar(node=None,
                               need_guard_check=False,
                               extract_code_at_start=[])
                ]
            })
            return

        raise NotImplementedError(func, args, kwargs)

    def gen_by_caller(self,
                      obj: Any,
                      var_type: Optional[type[Variable]] = None) -> bool:
        caller = self.caller
        while caller is not None:
            obj_in_caller = caller.state.objects.get_or_none(obj)
            if obj_in_caller is not None and (var_type is None or isinstance(
                    obj_in_caller, var_type)):
                return True
            caller = caller.caller
        return False

    def generic_jump_check(self) -> None:
        top_value = get_value_stack_from_top(self.frame, 0)
        if torch.is_tensor(top_value):
            raise ValueError("generic_jump TensorVariable() by tensor")
        if dyn.contains(top_value):
            raise ValueError("generic_jump TensorVariable() by dyn scalar")

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

    def BINARY_MATRIX_MULTIPLY(self, _inst: Instruction) -> None:
        self.binary_operation(operator.matmul)

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
        if torch.is_tensor(obj1):
            if torch.is_tensor(obj2) and obj2.dtype == torch.bool:
                raise ValueError("dynamic shape in tensor")
        self.call_function(operator.getitem, [obj1, obj2], {})

    def unary_operation(self, func: Callable[..., Any]) -> None:
        obj = get_value_stack_from_top(self.frame, 0)
        self.call_function(func, [obj], {})

    def UNARY_POSITIVE(self, _inst: Instruction) -> None:
        self.unary_operation(operator.pos)

    def UNARY_NEGATIVE(self, _inst: Instruction) -> None:
        self.unary_operation(operator.neg)

    def UNARY_NOT(self, _inst: Instruction) -> None:
        self.unary_operation(operator.not_)

    def UNARY_INVERT(self, _inst: Instruction) -> None:
        self.unary_operation(operator.invert)

    def COMPARE_OP(self, inst: Instruction) -> None:
        obj1 = get_value_stack_from_top(self.frame, 1)
        obj2 = get_value_stack_from_top(self.frame, 0)
        cmp_op = ('lt', 'le', 'eq', 'ne', 'gt', 'ge')
        self.call_function(getattr(operator, cmp_op[inst.arg]), [obj1, obj2],
                           {})

    def CONTAINS_OP(self, _inst: Instruction) -> None:
        self.binary_operation(operator.contains)

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

    def FORMAT_VALUE(self, _inst: Instruction) -> None:
        pass

    def BUILD_STRING(self, _inst: Instruction) -> None:
        pass

    def LOAD_CONST(self, _inst: Instruction) -> None:
        self.state.set_partial_var({
            -1: [
                PartialVar(node=None,
                           need_guard_check=False,
                           extract_code_at_start=[])
            ]
        })

    def SETUP_FINALLY(self, _inst: Instruction) -> None:
        pass

    def SETUP_WITH(self, _inst: Instruction) -> None:
        mgr = get_value_stack_from_top(self.frame, 0)
        if type(mgr) == torch.autograd.grad_mode.no_grad:
            self.call_function(mgr.__enter__, [], {})

    def JUMP_IF_NOT_EXC_MATCH(self, _inst: Instruction) -> None:
        pass

    # def YIELD_VALUE(self, _inst: Instruction) -> None:
    #     pass

    # def WITH_EXCEPT_START(self, _inst: Instruction) -> None:
    #     pass

    def RAISE_VARARGS(self, _inst: Instruction) -> None:
        pass

    def LOAD_FAST(self, inst: Instruction) -> None:
        if inst.argval not in self.state.stored_locals:
            obj = self.frame.f_locals[inst.argval]
            pos = StoreInLocal(inst.argval)
            if not self.state.objects.contains(obj):
                var = vs.make_var_from_value(
                    obj, True, self.state.objects.helper_functions,
                    self.state.fx_graph, [pos])
                self.state.add_object(var, obj)
            else:
                var = self.state.objects.get(obj)
                # remove pos loaded from function closure
                if len(var.extract_code_at_start) > 0:
                    old_pos = var.extract_code_at_start[0]
                    if isinstance(old_pos, StoreInAttr) and isinstance(
                            old_pos.self_pos, StoreInFreeVar):
                        var.clear_extract_code_at_start()
                        var.need_guard_check = True
                if var.prev is None:
                    var.add_extract_code_at_start(pos)

    def LOAD_DEREF(self, inst: Instruction) -> None:
        self.LOAD_FAST(inst)

    def LOAD_GLOBAL(self, inst: Instruction) -> None:
        if inst.argval not in self.state.stored_globals:
            if inst.argval in self.frame.f_globals:
                obj = self.frame.f_globals[inst.argval]
                store_pos: StorePos = StoreInGlobal(inst.argval)
            elif isinstance(
                    self.frame.f_globals['__builtins__'], dict
            ) and inst.argval in self.frame.f_globals['__builtins__']:
                obj = self.frame.f_globals['__builtins__'][inst.argval]
                store_pos = StoreInBuiltin(inst.argval, 'dict')
            elif hasattr(self.frame.f_globals['__builtins__'],
                         inst.argval):  # try first search in __builtins__
                obj = getattr(self.frame.f_globals['__builtins__'],
                              str(inst.argval))
                store_pos = StoreInBuiltin(inst.argval, 'attr')
            else:
                raise UnknownTypeError(inst.argval)

            var = vs.make_var_from_value(obj, True,
                                         self.state.objects.helper_functions,
                                         self.state.fx_graph, [store_pos])
            self.state.add_object(var, obj)

    def LOAD_METHOD(self, inst: Instruction) -> None:
        self_obj = get_value_stack_from_top(self.frame, 0)
        self.state.fetch_function_parameters(self_obj)
        self_var = self.state.objects.get(self_obj)
        is_bound = is_bound_method(self_obj, inst.argval)
        extract_code_at_start: list[StorePos] = [
            StoreInAttr(self_var.extract_code_at_start[0], id(self_obj),
                        inst.argval)
        ] if self_var.need_guard_check else []
        if is_bound:
            partial = [
                None,
                PartialVar(node=None,
                           need_guard_check=False,
                           extract_code_at_start=extract_code_at_start)
            ]
        else:
            partial = [
                PartialVar(node=None,
                           need_guard_check=False,
                           extract_code_at_start=extract_code_at_start), None
            ]
        self.state.set_partial_var({-1: partial})

    def LOAD_ATTR(self, inst: Instruction) -> None:
        obj = get_value_stack_from_top(self.frame, 0)
        attr = getattr(obj, inst.argval)
        obj_var = self.state.objects.get(obj)
        new_extract: list[StorePos] = [
            StoreInAttr(pos, id(obj), inst.argval)
            for pos in obj_var.extract_code_at_start
        ]

        if inst.argval in obj_var.modified_attrs:
            return

        need_guard_check = obj_var.need_guard_check
        if id(obj) == id(self.state.varargs) and inst.argval in dir(tuple):
            need_guard_check = False
        if id(obj) == id(self.state.varkw) and inst.argval in dir(dict):
            need_guard_check = False
        node1 = None
        if isinstance(obj, torch.Tensor) and isinstance(attr, torch.Tensor):
            if isinstance(obj_var, vs.TorchParamVar):
                if obj not in self.state.subparam_paths:
                    self.state.add_subparam(obj)
                node: Optional[torch.fx.Node] = self.state.fx_graph.create_node(
                    "get_attr", self.state.subparam_paths[obj], (), {})
                if inst.argval != 'data':
                    node1 = self.state.fx_graph.create_node(
                        "call_function", getattr, (node, inst.argval), {})
            elif inst.argval == 'data':
                node = obj_var.as_fx_node()
            else:
                node = self.state.fx_graph.create_node(
                    "call_function", getattr,
                    (obj_var.as_fx_node(), inst.argval), {})
        elif config.get_config('dynshape') and isinstance(
                obj, torch.Tensor) and inst.argval == 'shape':
            node = self.state.fx_graph.create_node("call_method", "size",
                                                   (obj_var.as_fx_node(),), {})
            need_guard_check = False
        elif isinstance(obj, torch.Tensor) and inst.argval == 'data':
            node = obj_var.as_fx_node()
        else:
            node = None
        if node1 is not None:
            node = node1
        partial: list[Optional[PartialVar]] = [
            PartialVar(node=node,
                       need_guard_check=need_guard_check,
                       extract_code_at_start=new_extract)
        ]

        self.state.set_partial_var({-1: partial})

    def LOAD_CLOSURE(self, inst: Instruction) -> None:
        # if inst.argval not in self.state.stored_locals:
        obj = get_from_freevars(self.frame, inst.arg)
        pos = StoreInFreeVar(inst.arg)
        cell_obj = parse_cell(obj)
        need_guard_check = not isinstance(
            cell_obj, NullObject) and not self.state.objects.contains(cell_obj)
        var = vs.make_var_from_value(obj, need_guard_check,
                                     self.state.objects.helper_functions,
                                     self.state.fx_graph, [pos])
        self.state.add_object(var, obj)

    def CALL_FUNCTION(self, inst: Instruction) -> None:
        num_args = inst.argval
        args = [
            get_value_stack_from_top(self.frame, i)
            for i in range(num_args - 1, -1, -1)
        ]
        kwargs: dict[str, Any] = {}
        func = get_value_stack_from_top(self.frame, num_args)
        # print(f"function: {func}, type: {type(func)},args:{args}, kwargs:{kwargs}")
        for i, obj in enumerate(itertools.chain(args, kwargs.values())):
            self.state.fetch_function_parameters(obj)
        self.call_function(func, args, kwargs)

    def CALL_METHOD(self, inst: Instruction) -> None:
        num_args = inst.argval
        args = [
            get_value_stack_from_top(self.frame, i)
            for i in range(num_args - 1, -1, -1)
        ]
        kwargs: dict[str, Any] = {}
        for i, obj in enumerate(itertools.chain(args, kwargs.values())):
            self.state.fetch_function_parameters(obj)
        self_val = get_value_stack_from_top(self.frame, num_args)
        meth_val = get_value_stack_from_top(self.frame, num_args + 1)
        if isinstance(meth_val, NullObject):
            # Stack layout: ... | NULL | callable | arg1 | ... | argN
            # print(f"call method: {self_val}, type: {type(self_val)},args:{args}, kwargs:{kwargs}")
            self.call_function(self_val, args, kwargs)
        else:
            # Stack layout: ... | method | self | arg1 | ... | argN
            # print(f"call method: {meth_val}, type: {type(meth_val)},args:{[self_val] + args}, kwargs:{kwargs}")
            self.call_function(meth_val, [self_val] + args, kwargs)

    def CALL_FUNCTION_KW(self, inst: Instruction) -> None:
        num_args = inst.argval
        args = [
            get_value_stack_from_top(self.frame, i + 1)
            for i in range(num_args - 1, -1, -1)
        ]
        # for xx in args: print("kwarg", id(xx), xx)
        kw_names = get_value_stack_from_top(self.frame, 0)
        func = get_value_stack_from_top(self.frame, num_args + 1)
        kwargs: dict[str, Any] = {}
        for arg, kw_name in zip(args[-len(kw_names):], kw_names):
            kwargs[kw_name] = arg
        self.state.fetch_function_parameters(kwargs)
        args = args[:-len(kw_names)]
        if hasattr(func,
                   '__self__') and func.__self__ is not None and not isinstance(
                       func.__self__, ModuleType):
            args = [func.__self__] + list(args)
        for i, obj in enumerate(itertools.chain(args, kwargs.values())):
            self.state.fetch_function_parameters(obj)
        self.call_function(func, args, kwargs)

    def CALL_FUNCTION_EX(self, inst: Instruction) -> None:
        offset = inst.argval & 1
        func = get_value_stack_from_top(self.frame, 1 + offset)
        args = get_value_stack_from_top(self.frame, offset)
        if offset == 1:
            kwargs = get_value_stack_from_top(self.frame, 0)
        else:
            kwargs = {}
        # print(f"function ex: {func}, type: {type(func)},args:{(func.__self__,) + args}, kwargs:{kwargs}")
        if hasattr(func,
                   '__self__') and func.__self__ is not None and not isinstance(
                       func.__self__, ModuleType):
            args = [func.__self__] + list(args)
        if not isinstance(args, torch.Tensor):  # call(*x)
            for i, obj in enumerate(itertools.chain(args, kwargs.values())):
                self.state.fetch_function_parameters(obj)
        self.call_function(func, args, kwargs)

    def STORE_FAST(self, inst: Instruction) -> None:
        self.state.add_stored_locals(inst.argval)

    def STORE_DEREF(self, inst: Instruction) -> None:
        self.state.add_stored_locals(inst.argval)

    def STORE_SUBSCR(self, inst: Instruction) -> None:
        index = get_value_stack_from_top(self.frame, 0)
        target = get_value_stack_from_top(self.frame, 1)
        value = get_value_stack_from_top(self.frame, 2)
        if isinstance(target, torch.Tensor):
            # still use the original node, so no need to update object table
            self.state.record_function(operator.setitem, [target, index, value],
                                       {},
                                       add_partial_var=False)
        else:
            self.state.add_inplace_update_obj(target)

    def DELETE_SUBSCR(self, inst: Instruction) -> None:
        index = get_value_stack_from_top(self.frame, 0)
        target = get_value_stack_from_top(self.frame, 1)
        if isinstance(target, torch.Tensor):
            self.state.record_function(operator.delitem, [target, index], {},
                                       add_partial_var=False)
        else:
            self.state.add_inplace_update_obj(target)

    def STORE_ATTR(self, inst: Instruction) -> None:
        value = get_value_stack_from_top(self.frame, 1)
        self_obj = get_value_stack_from_top(self.frame, 0)
        self.state.fetch_function_parameters(self_obj)
        self.state.fetch_function_parameters(value)
        value_var = self.state.objects.get(value)
        self_obj_var = self.state.objects.get(self_obj)
        store_pos: list[StorePos] = []
        node: Optional[torch.fx.Node] = None
        if isinstance(self_obj, (torch.Tensor, torch.nn.Module)):
            store_pos = [pos for pos in self_obj_var.extract_code_at_start]
        if len(store_pos) == 0 and isinstance(self_obj, torch.Tensor):
            node = self_obj_var.as_fx_node()
        if node is None:
            new_self_var = vs.make_var_from_value(
                self_obj, False, self.state.objects.helper_functions,
                self.state.fx_graph, store_pos)
        else:
            new_self_var = vs.TensorVar.from_tensor_and_node(
                self_obj, node, False, store_pos)
        new_self_var.add_modified_attr(inst.argval, value_var)
        self.state.objects.update_by_id(new_self_var, id(self_obj))

    def DELETE_ATTR(self, inst: Instruction) -> None:
        pass

    def DELETE_FAST(self, inst: Instruction) -> None:
        if inst.argval in self.state.stored_locals:
            self.state.delete_stored_locals(inst.argval)

    def IS_OP(self, inst: Instruction) -> None:
        self.binary_operation(operator.is_)

    def BUILD_TUPLE(self, inst: Instruction) -> None:
        partial: list[Optional[PartialVar]] = [
            PartialVar(node=None,
                       need_guard_check=False,
                       extract_code_at_start=[])
        ]
        self.state.set_partial_var({-1: partial})

    def BUILD_LIST(self, inst: Instruction) -> None:
        partial: list[Optional[PartialVar]] = [
            PartialVar(node=None,
                       need_guard_check=False,
                       extract_code_at_start=[])
        ]
        self.state.set_partial_var({-1: partial})

    def BUILD_SET(self, inst: Instruction) -> None:
        pass

    def BUILD_CONST_KEY_MAP(self, inst: Instruction) -> None:
        pass

    def BUILD_MAP(self, inst: Instruction) -> None:
        pass

    def LIST_TO_TUPLE(self, inst: Instruction) -> None:
        pass

    def LIST_EXTEND(self, inst: Instruction) -> None:
        pass

    def LIST_APPEND(self, inst: Instruction) -> None:
        pass

    def MAP_ADD(self, inst: Instruction) -> None:
        pass

    def DICT_MERGE(self, inst: Instruction) -> None:
        pass

    def DICT_UPDATE(self, inst: Instruction) -> None:
        pass

    def IMPORT_NAME(self, inst: Instruction) -> None:
        partial = [
            PartialVar(node=None,
                       need_guard_check=False,
                       extract_code_at_start=[]), None
        ]
        self.state.set_partial_var({-1: partial})
        pass

    def IMPORT_FROM(self, inst: Instruction) -> None:
        partial = [
            PartialVar(node=None,
                       need_guard_check=False,
                       extract_code_at_start=[]), None
        ]
        self.state.set_partial_var({-1: partial})
        pass

    def UNPACK_SEQUENCE(self, inst: Instruction) -> None:
        seq = get_value_stack_from_top(self.frame, 0)
        if isinstance(
                seq,
            (tuple, list, torch.Size, torch.Tensor, torch.nn.ModuleList)):
            node = None
            if isinstance(seq, torch.Tensor):
                partials: list[Optional[PartialVar]] = []
                var = self.state.objects.get(seq)
                node = var.as_fx_node()
                for i in range(len(seq)):
                    fx_node = self.state.fx_graph.create_node(
                        "call_function", operator.getitem, (node, i), {})
                    partial = PartialVar(node=fx_node,
                                         need_guard_check=False,
                                         extract_code_at_start=[],
                                         make_var_fn=vs.make_var_from_value)
                    partials.append(partial)
                self.state.set_partial_var({-1: partials})
            else:
                self.state.set_partial_var({
                    -1: [
                        PartialVar(node=node,
                                   need_guard_check=False,
                                   extract_code_at_start=[],
                                   make_var_fn=vs.make_var_from_value)
                        for _ in range(len(seq))
                    ]
                })
        else:
            # unpack generator object
            # self.state.set_partial_var({
            #     -1: [
            #         PartialVar(node=None,
            #                     need_guard_check=False,
            #                     extract_code_at_start=[],
            #                     make_var_fn=vs.make_var_from_value)
            #         for _ in seq
            #     ]
            # })
            # pass
            print("check data", seq, type(seq))
            if self.state.objects.contains(seq):
                print("jjjjjj")
            for i in seq:
                print(i)
            raise NotImplementedError

    def UNPACK_EX(self, inst: Instruction) -> None:
        seq = get_value_stack_from_top(self.frame, 0)
        if isinstance(seq, (tuple, list, torch.Size)):
            self.state.set_partial_var({
                -1: [
                    PartialVar(node=None,
                               need_guard_check=False,
                               extract_code_at_start=[],
                               make_var_fn=vs.make_var_from_value)
                    for _ in range(len(seq))
                ]
            })
        elif isinstance(seq, torch.Tensor):
            self.state.set_partial_var({
                -1: [
                    PartialVar(node=None,
                               need_guard_check=False,
                               extract_code_at_start=[],
                               make_var_fn=vs.make_var_from_value)
                    for _ in range(len(seq))
                ]
            })
            self.call_function(iter, [seq], {})
        else:
            raise ValueError("unknow type in unpack_ex", type(seq))

    def POP_TOP(self, _inst: Instruction) -> None:
        pass

    def POP_BLOCK(self, _inst: Instruction) -> None:
        pass

    def POP_EXCEPT(self, _inst: Instruction) -> None:
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
        self.generic_jump_check()

    def POP_JUMP_IF_TRUE(self, _inst: Instruction) -> None:
        self.generic_jump_check()

    def JUMP_IF_TRUE_OR_POP(self, _inst: Instruction) -> None:
        self.generic_jump_check()

    def JUMP_IF_FALSE_OR_POP(self, _inst: Instruction) -> None:
        self.generic_jump_check()

    def JUMP_FORWARD(self, inst: Instruction) -> None:
        pass

    def JUMP_ABSOLUTE(self, inst: Instruction) -> None:
        pass

    def EXTENDED_ARG(self, inst: Instruction) -> None:
        pass

    def GET_ITER(self, _inst: Instruction) -> None:
        obj = get_value_stack_from_top(self.frame, 0)
        self.state.fetch_function_parameters(obj)
        obj_var = self.state.objects.get(obj)
        extract_code_at_start: list[StorePos] = [
            ExtractFromMethod(pos, id(obj), '__iter__')
            for pos in obj_var.extract_code_at_start
        ]

        def make_iterable_fn(
                value: Any, need_guard_check: bool,
                _helper_functions: vs.HelperFunctions,
                fx_graph: Optional[FxGraph],
                extract_code_at_start: Optional[list[StorePos]]) -> vs.Variable:
            if extract_code_at_start is None:
                extract_code_at_start = []
            return vs.IteratorVar.from_parent_var(value, obj_var, id(obj), 0,
                                                  need_guard_check,
                                                  extract_code_at_start)

        make_var_fn: Optional[
            MAKE_VAR_FN_TYPE] = make_iterable_fn if not isinstance(
                obj, range) else None

        self.state.set_partial_var({
            -1: [
                PartialVar(node=None,
                           need_guard_check=False,
                           extract_code_at_start=extract_code_at_start,
                           make_var_fn=make_var_fn)
            ]
        })

        if isinstance(obj, torch.Tensor):
            self.call_function(iter, [obj], {})

    def FOR_ITER(self, _original_inst: Instruction) -> None:
        original_pc, original_inst = self.code.get_orig_inst(self.frame.f_lasti)
        guard_pc = self.frame.f_lasti // 2
        while self.code.guard_insts[guard_pc].opname == "EXTENDED_ARG":
            guard_pc += 1
        guard_inst = self.code.guard_insts[guard_pc]
        iterator = get_value_stack_from_top(self.frame, 0)
        is_dynamic = dyn.contains_pc(self.frame_id,
                                     original_pc)  # TODO: remove hardcode
        if is_dynamic:
            end_pc_original = end_of_control_flow(self.code.original_insts,
                                                  original_pc)
            end_pc_guard = end_of_control_flow(self.code.guard_insts,
                                               self.frame.f_lasti // 2)
            num_iter_var = self.state.objects.get(iterator)
            if isinstance(num_iter_var, vs.RangeIterVar):
                num_iter = num_iter_var.len
            else:
                raise NotImplementedError
            if self.code.is_match(
                    end_pc_original,
                    end_pc_guard):  # have graph break in control flow
                stack_objs = get_all_objects_in_stack(self.frame)
                self.state.mark_defer_restart(
                    DeferRestartState(stack_objs, self.get_live_objs(),
                                      self.frame.f_lasti, f"dynamic for_iter",
                                      False))
                dyn.pop_dynamic_pc(self.frame_id, original_pc)
                new_tracker = push_tracker(self.frame,
                                           self.frame_id,
                                           read_stack=True,
                                           cf_info=ForLoopInfo(
                                               start_pc=original_pc,
                                               end_pc=end_pc_original,
                                               num_iter=num_iter))
                for obj in stack_objs:
                    var = self.state.objects.get_or_none(obj)
                    if var is not None:
                        new_tracker.state.objects.add(var, obj)
                new_tracker.record(self.frame, self.frame_id)
                return
            else:
                raise NotImplementedError("orz")
        else:
            obj = get_value_stack_from_top(self.frame, 0)
            obj_var = self.state.objects.get(obj)
            if isinstance(obj_var, vs.AnyVar):
                if self.gen_by_caller(obj, vs.IteratorVar):
                    obj_var = vs.IteratorVar.from_parent_var(
                        obj, None, id(obj), 0, False, [])
                    self.state.add_object(obj_var, obj)
                    obj_var.need_guard_check = False
                    self.state.mark_cannot_guard()
            assert isinstance(obj_var, vs.IteratorVar) or isinstance(
                obj, type(range(0).__iter__()))
            normal_pc = guard_pc + 1
            guard_target = guard_inst.target
            assert guard_target is not None
            end_loop_pc = self.code.get_pc_by_inst(guard_target)

            def make_iterable_fn(
                    value: Any, need_guard_check: bool,
                    _helper_functions: vs.HelperFunctions,
                    _fx_graph: Optional[FxGraph],
                    extract_code_at_start: Optional[list[StorePos]]
            ) -> vs.Variable:
                assert isinstance(obj_var, vs.IteratorVar)
                if extract_code_at_start is None:
                    extract_code_at_start = []
                return vs.IteratorVar.from_parent_var(value, obj_var.parent_var,
                                                      obj_var.parent_idx,
                                                      obj_var.num_iters + 1,
                                                      need_guard_check,
                                                      extract_code_at_start)

            def make_dynamic_input_fn(
                    value: Any, need_guard_check: bool,
                    helper_functions: vs.HelperFunctions,
                    fx_graph: Optional[FxGraph],
                    extract_code_at_start: Optional[list[StorePos]]
            ) -> vs.Variable:
                assert is_scalar(value)
                dyn.mark_dynamic(value, dyn.ScalarWithUnknownValue())
                if extract_code_at_start is None:
                    extract_code_at_start = []
                var = helper_functions.get_or_make_var(value, need_guard_check,
                                                       fx_graph,
                                                       extract_code_at_start)
                return var

            make_iter_fn = make_iterable_fn if not isinstance(
                obj, type(range(0).__iter__())) else None

            if self.cf_info is not None and self.cf_info.start_pc == original_pc:
                loop_info = self.cf_info
                assert isinstance(loop_info, ForLoopInfo)
                if loop_info.cur_iter == 1:
                    obj_var.get_oldest_var().disable_guard_check()
                    self.commit_loop_subgraph()
                make_value_fn = make_dynamic_input_fn if self.cf_info is not None and self.cf_info.start_pc == original_pc else None
                self.state.set_partial_var({
                    normal_pc: [
                        PartialVar(node=None,
                                   need_guard_check=False,
                                   extract_code_at_start=[IterValue()],
                                   make_var_fn=make_value_fn),
                        PartialVar(node=None,
                                   need_guard_check=False,
                                   extract_code_at_start=[],
                                   inplace_ref=obj,
                                   make_var_fn=make_iter_fn)
                    ],
                    end_loop_pc: []
                })
                loop_info.cur_iter += 1
                return

            self.state.set_partial_var({
                normal_pc: [
                    PartialVar(node=None,
                               need_guard_check=False,
                               extract_code_at_start=[]),
                    PartialVar(node=None,
                               need_guard_check=False,
                               extract_code_at_start=[],
                               inplace_ref=obj,
                               make_var_fn=make_iter_fn)
                ],
                end_loop_pc: []
            })

    def MAKE_FUNCTION(self, _inst: Instruction) -> None:
        self.state.set_partial_var({
            -1: [
                PartialVar(node=None,
                           need_guard_check=False,
                           extract_code_at_start=[])
            ]
        })


trackers: list[GuardTracker] = []


def push_tracker(frame: FrameType,
                 frame_id: int,
                 read_stack: bool = False,
                 cf_info: Optional[ControlFlowInfo] = None) -> GuardTracker:
    if len(trackers) > 0:
        caller = trackers[-1]
    else:
        caller = None
    new_tracker = GuardTracker(frame, frame_id, caller, read_stack, cf_info)
    trackers.append(new_tracker)
    print("push tracker", frame_id, "frame", hex(id(frame)),
          "frame_id", frame_id, "read_stack", read_stack, "cf_info",
          type(cf_info), "all", [t.frame_id for t in trackers])
    return new_tracker


def pop_tracker(frame_id: int) -> None:
    print("before pop_tracker", [t.frame_id for t in trackers], "frame_id",
          frame_id)
    to_pop = trackers.pop()
    assert to_pop.frame_id == frame_id
    assert to_pop.state.is_empty


def record(frame: FrameType, frame_id: int) -> None:
    if id(frame) != id(trackers[-1].frame):
        if trackers[-1].state.calling_func is not None:
            print("push tracker due to record")
            push_tracker(frame, frame_id)
    trackers[-1].record(frame, frame_id)


def reset() -> None:
    trackers.clear()
