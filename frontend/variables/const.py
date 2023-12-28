from typing import TYPE_CHECKING, Union, Optional, Callable, Any
from frontend.pycode_generator import GraphFnCodegen

import torch.fx
from types import ModuleType
from enum import Enum
from .base import Variable, HelperFunctions
from ..pycode_writer import get_float_string
from ..fx_graph import NodeArgs, FxGraph
from ..utils import NullObject, null_object
from ..store_pos import StorePos, StoreInFreeVar, StoreInAttr
from ..c_api import parse_cell
if TYPE_CHECKING:
    from ..pycode_generator import GraphFnCodegen, GuardFnCodegen
    from ..object_table import ObjectTable


class NoneVar(Variable):

    def __init__(self, need_guard_check: bool, obj: None,
                 extract_code_at_start: list[StorePos]) -> None:
        super().__init__(need_guard_check, obj, extract_code_at_start)

    def make_guard_inner(self, codegen: "GuardFnCodegen",
                         pos: StorePos) -> None:
        for pos in self.extract_code_at_start:
            codegen.add_check((f"{pos} is None", pos))

    def make_output_inner(self, name_in_graph_fn: str, store_pos: StorePos,
                          codegen: "GraphFnCodegen", in_return: bool,
                          idx: int) -> None:
        codegen.output(name_in_graph_fn, store_pos, "None", in_return, idx)

    @classmethod
    def from_value(cls, value: None, need_guard_check: bool,
                   _helper_functions: HelperFunctions,
                   _fx_graph: Optional[FxGraph],
                   extract_code_at_start: list[StorePos]) -> "NoneVar":
        return cls(need_guard_check, value, extract_code_at_start)

    def as_fx_node(self) -> NodeArgs:
        return None


class NullVar(Variable):

    def __init__(self, need_guard_check: bool, obj: NullObject,
                 extract_code_at_start: list[StorePos]) -> None:
        super().__init__(need_guard_check, obj, extract_code_at_start)

    def make_guard_inner(self, codegen: "GuardFnCodegen",
                         pos: StorePos) -> None:
        pass

    def make_output_inner(self, name_in_graph_fn: str, store_pos: StorePos,
                          codegen: "GraphFnCodegen", in_return: bool,
                          idx: int) -> None:
        name_in_codegen = codegen.add_obj(null_object, "NULL_VAR")
        codegen.output(name_in_graph_fn, store_pos, f"{name_in_codegen} # NULL",
                       in_return, idx)

    @classmethod
    def from_value(cls, value: NullObject, need_guard_check: bool,
                   _helper_functions: HelperFunctions,
                   _fx_graph: Optional[FxGraph],
                   extract_code_at_start: list[StorePos]) -> "NullVar":
        return cls(need_guard_check, value, extract_code_at_start)

    def as_fx_node(self) -> NodeArgs:
        raise NotImplementedError()


class CodeVar(Variable):

    def __init__(self, need_guard_check: bool, obj: None,
                 extract_code_at_start: list[StorePos]) -> None:
        super().__init__(need_guard_check, obj, extract_code_at_start)

    def make_guard_inner(self, codegen: "GuardFnCodegen",
                         pos: StorePos) -> None:
        codegen.add_id_check((f"id({pos}) == {id(self.obj)}", pos), self.obj)

    def make_output_inner(self, name_in_graph_fn: str, store_pos: StorePos,
                          codegen: "GraphFnCodegen", in_return: bool,
                          idx: int) -> None:
        name = codegen.add_obj(self.obj, "CODE_VAR")
        codegen.output(name_in_graph_fn, store_pos, name, in_return, idx)

    @classmethod
    def from_value(cls, value: None, need_guard_check: bool,
                   _helper_functions: HelperFunctions,
                   _fx_graph: Optional[FxGraph],
                   extract_code_at_start: list[StorePos]) -> "CodeVar":
        return cls(need_guard_check, value, extract_code_at_start)

    def as_fx_node(self) -> NodeArgs:
        return None


class SliceVar(Variable):
    start: Optional[int]
    stop: Optional[int]
    step: Optional[int]

    def __init__(self, start: Optional[int], stop: Optional[int],
                 step: Optional[int], need_guard_check: bool, obj: slice,
                 extract_code_at_start: list[StorePos]) -> None:
        super().__init__(need_guard_check, obj, extract_code_at_start)
        self.start = start
        self.stop = stop
        self.step = step

    def make_guard_inner(self, codegen: "GuardFnCodegen",
                         pos: StorePos) -> None:
        codegen.add_check(
            (f"{pos} == slice({self.start}, {self.stop}, {self.step})", pos))

    def make_output_inner(self, name_in_graph_fn: str, store_pos: StorePos,
                          codegen: "GraphFnCodegen", in_return: bool,
                          idx: int) -> None:
        codegen.output(name_in_graph_fn, store_pos,
                       f"slice({self.start}, {self.stop}, {self.step})",
                       in_return, idx)

    @classmethod
    def from_value(cls, value: slice, need_guard_check: bool,
                   _helper_functions: HelperFunctions,
                   _fx_graph: Optional[FxGraph],
                   extract_code_at_start: list[StorePos]) -> "SliceVar":
        return cls(value.start, value.stop, value.step, need_guard_check, value,
                   extract_code_at_start)

    def as_fx_node(self) -> NodeArgs:
        return slice(self.start, self.stop, self.step)


class EllipsisVar(Variable):

    def __init__(self, need_guard_check: bool, obj: Any,
                 extract_code_at_start: list[StorePos]) -> None:
        super().__init__(need_guard_check, obj, extract_code_at_start)

    def make_guard_inner(self, codegen: "GuardFnCodegen",
                         pos: StorePos) -> None:
        codegen.add_id_check((f"id({pos}) == {id(self.obj)}", pos), self.obj)

    def make_output_inner(self, name_in_graph_fn: str, store_pos: StorePos,
                          codegen: "GraphFnCodegen", in_return: bool,
                          idx: int) -> None:
        name = codegen.add_obj(self.obj, "Ellipsis_VAR")
        codegen.output(name_in_graph_fn, store_pos, name, in_return, idx)

    @classmethod
    def from_value(cls, value: Any, need_guard_check: bool,
                   _helper_functions: HelperFunctions,
                   _fx_graph: Optional[FxGraph],
                   extract_code_at_start: list[StorePos]) -> "EllipsisVar":
        return cls(need_guard_check, value, extract_code_at_start)

    def as_fx_node(self) -> NodeArgs:
        return Ellipsis


torch_modules = set([torch])


class ModuleVar(Variable):

    def __init__(self, module: ModuleType, need_guard_check: bool,
                 extract_code_at_start: list[StorePos]) -> None:
        super().__init__(need_guard_check, module, extract_code_at_start)

    def make_guard_inner(self, codegen: "GuardFnCodegen",
                         pos: StorePos) -> None:
        codegen.add_id_check((f"id({pos}) == {id(self.obj)}", pos), self.obj)

    def make_output_inner(self, name_in_graph_fn: str, store_pos: StorePos,
                          codegen: "GraphFnCodegen", in_return: bool,
                          idx: int) -> None:
        name_in_codegen = codegen.add_obj(self.obj,
                                          f"MODULE_{self.obj.__name__}")
        codegen.output(name_in_graph_fn, store_pos, name_in_codegen, in_return,
                       idx)

    @classmethod
    def from_value(cls, value: ModuleType, need_guard_check: bool,
                   _helper_functions: HelperFunctions,
                   _fx_graph: Optional[FxGraph],
                   extract_code_at_start: list[StorePos]) -> "ModuleVar":
        return cls(value, need_guard_check, extract_code_at_start)


class FunctionVar(Variable):
    closure_vars: list[Variable]
    obj_ids: list[int]

    def __init__(self, func: Callable[..., Any], need_guard_check: bool,
                 helper_functions: HelperFunctions, fx_graph: Optional[FxGraph],
                 extract_code_at_start: list[StorePos]) -> None:
        super().__init__(need_guard_check, func, extract_code_at_start)
        self.closure_vars = []
        self.obj_ids = []
        if hasattr(func, "__code__") and hasattr(func, "__closure__"):
            if func.__closure__ is not None:
                assert len(func.__code__.co_freevars) == len(func.__closure__)
                for i, x in enumerate(func.__closure__):
                    if parse_cell(x) != func:
                        cell_var = helper_functions.get_or_make_var(
                            x, need_guard_check, fx_graph, [StoreInFreeVar(i)])
                        self.closure_vars.append(cell_var)
                        self.obj_ids.append(id(x))

    def make_guard_inner(self, codegen: "GuardFnCodegen",
                         pos: StorePos) -> None:
        if hasattr(self.obj, '__self__') and isinstance(self.obj.__self__,
                                                        torch.Tensor):
            pass
        else:
            codegen.add_id_check((f"id({pos}) == {id(self.obj)}", pos),
                                 self.obj)

    def make_output_inner(self, name_in_graph_fn: str, store_pos: StorePos,
                          codegen: "GraphFnCodegen", in_return: bool,
                          idx: int) -> None:
        name_in_codegen = codegen.add_obj(self.obj, f"_{self.obj.__name__}")
        codegen.output(name_in_graph_fn, store_pos, name_in_codegen, in_return,
                       idx)

    @classmethod
    def from_value(cls, value: Callable[..., Any], need_guard_check: bool,
                   helper_functions: HelperFunctions,
                   fx_graph: Optional[FxGraph],
                   extract_code_at_start: list[StorePos]) -> "FunctionVar":
        return cls(value, need_guard_check, helper_functions, fx_graph,
                   extract_code_at_start)

    def add_subvars_to_table(self, table: 'ObjectTable') -> None:
        for i, (var, idx) in enumerate(zip(self.closure_vars, self.obj_ids)):
            old_var = table.get_or_none_by_id(idx)
            if old_var is not None:
                new_extract: list[StorePos] = [StoreInFreeVar(i)]
                old_var.extract_code_at_start.extend(new_extract)
                old_var.need_guard_check |= self.need_guard_check
            else:
                table.add_by_id(var, idx)
                var.add_subvars_to_table(table)

    # def as_fx_node(self) -> NodeArgs:
    #     return self.obj


class RangeVar(Variable):
    start: int
    stop: int
    step: int

    def __init__(self, start: int, stop: int, step: int, need_guard_check: bool,
                 obj: range, extract_code_at_start: list[StorePos]) -> None:
        super().__init__(need_guard_check, obj, extract_code_at_start)
        self.start = start
        self.stop = stop
        self.step = step

    def make_guard_inner(self, codegen: "GuardFnCodegen",
                         pos: StorePos) -> None:
        codegen.add_check(
            (f"{pos} == range({self.start}, {self.stop}, {self.step})", pos))

    def make_output_inner(self, name_in_graph_fn: str, store_pos: StorePos,
                          codegen: "GraphFnCodegen", in_return: bool,
                          idx: int) -> None:
        codegen.output(name_in_graph_fn, store_pos,
                       f"range({self.start}, {self.stop}, {self.step})",
                       in_return, idx)

    @classmethod
    def from_value(cls, value: range, need_guard_check: bool,
                   _helper_functions: HelperFunctions,
                   _fx_graph: Optional[FxGraph],
                   extract_code_at_start: list[StorePos]) -> "RangeVar":
        return cls(value.start, value.stop, value.step, need_guard_check, value,
                   extract_code_at_start)

    def as_fx_node(self) -> NodeArgs:
        return range(self.start, self.stop, self.step)
