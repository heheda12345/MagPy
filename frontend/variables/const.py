from typing import TYPE_CHECKING, Union, Optional, Callable, Any
from frontend.pycode_generator import GraphFnCodegen

import torch.fx
from types import ModuleType
from enum import Enum
from .base import Variable
from ..pycode_writer import get_float_string
from ..fx_graph import NodeArgs, FxGraph
from ..utils import NullObject, null_object
from ..store_pos import StorePos
if TYPE_CHECKING:
    from ..pycode_generator import GraphFnCodegen, GuardFnCodegen


class NoneVar(Variable):

    def __init__(self,
                 need_guard_check: bool,
                 obj: None,
                 extract_code_at_start: list[StorePos] = []) -> None:
        super().__init__(need_guard_check, obj, extract_code_at_start)

    def make_guard_inner(self, codegen: "GuardFnCodegen",
                         pos: StorePos) -> None:
        for pos in self.extract_code_at_start:
            codegen.add_check(f"{pos} is None")

    def make_output_inner(self, name_in_graph_fn: str, store_pos: StorePos,
                          codegen: "GraphFnCodegen", in_return: bool,
                          idx: int) -> None:
        codegen.output(name_in_graph_fn, store_pos, "None", in_return, idx)

    @classmethod
    def from_value(cls,
                   value: None,
                   need_guard_check: bool,
                   _get_or_make_var: Callable[
                       [Any, bool, Optional[FxGraph], list[StorePos]],
                       Variable],
                   _fx_graph: Optional[FxGraph] = None,
                   extract_code_at_start: list[StorePos] = []) -> "NoneVar":
        return cls(need_guard_check, value, extract_code_at_start)

    def as_fx_node(self) -> NodeArgs:
        return None


class NullVar(Variable):

    def __init__(self,
                 need_guard_check: bool,
                 obj: NullObject,
                 extract_code_at_start: list[StorePos] = []) -> None:
        super().__init__(need_guard_check, obj, extract_code_at_start)

    def make_guard_inner(self, codegen: "GuardFnCodegen",
                         pos: StorePos) -> None:
        pass

    def make_output_inner(self, name_in_graph_fn: str, store_pos: StorePos,
                          codegen: "GraphFnCodegen", in_return: bool,
                          idx: int) -> None:
        name_in_codegen = codegen.add_var(null_object, "NULL_VAR")
        codegen.output(name_in_graph_fn, store_pos, f"{name_in_codegen} # NULL",
                       in_return, idx)

    @classmethod
    def from_value(cls,
                   value: NullObject,
                   need_guard_check: bool,
                   _get_or_make_var: Callable[
                       [Any, bool, Optional[FxGraph], list[StorePos]],
                       Variable],
                   _fx_graph: Optional[FxGraph] = None,
                   extract_code_at_start: list[StorePos] = []) -> "NullVar":
        return cls(need_guard_check, value, extract_code_at_start)

    def as_fx_node(self) -> NodeArgs:
        raise NotImplementedError()


class SliceVar(Variable):
    start: Optional[int]
    stop: Optional[int]
    step: Optional[int]

    def __init__(self,
                 start: Optional[int],
                 stop: Optional[int],
                 step: Optional[int],
                 need_guard_check: bool,
                 obj: slice,
                 extract_code_at_start: list[StorePos] = []) -> None:
        super().__init__(need_guard_check, obj, extract_code_at_start)
        self.start = start
        self.stop = stop
        self.step = step

    def make_guard_inner(self, codegen: "GuardFnCodegen",
                         pos: StorePos) -> None:
        codegen.add_check(
            f"{pos} == slice({self.start}, {self.stop}, {self.step})")

    def make_output_inner(self, name_in_graph_fn: str, store_pos: StorePos,
                          codegen: "GraphFnCodegen", in_return: bool,
                          idx: int) -> None:
        codegen.output(name_in_graph_fn, store_pos,
                       f"slice({self.start}, {self.stop}, {self.step})",
                       in_return, idx)

    @classmethod
    def from_value(cls,
                   value: slice,
                   need_guard_check: bool,
                   _get_or_make_var: Callable[
                       [Any, bool, Optional[FxGraph], list[StorePos]],
                       Variable],
                   _fx_graph: Optional[FxGraph] = None,
                   extract_code_at_start: list[StorePos] = []) -> "SliceVar":
        return cls(value.start, value.stop, value.step, need_guard_check, value,
                   extract_code_at_start)

    def as_fx_node(self) -> NodeArgs:
        return slice(self.start, self.stop, self.step)


torch_modules = set([torch])


class ModuleVar(Variable):

    def __init__(self,
                 module: ModuleType,
                 need_guard_check: bool,
                 extract_code_at_start: list[StorePos] = []) -> None:
        super().__init__(need_guard_check, module, extract_code_at_start)

    def make_guard_inner(self, codegen: "GuardFnCodegen",
                         pos: StorePos) -> None:
        codegen.add_id_check(f"id({pos}) == {id(self.obj)}", self.obj)

    def make_output_inner(self, name_in_graph_fn: str, store_pos: StorePos,
                          codegen: "GraphFnCodegen", in_return: bool,
                          idx: int) -> None:
        name_in_codegen = codegen.add_var(self.obj,
                                          f"MODULE_{self.obj.__name__}")
        codegen.output(name_in_graph_fn, store_pos, name_in_codegen, in_return,
                       idx)

    @classmethod
    def from_value(cls,
                   value: ModuleType,
                   need_guard_check: bool,
                   _get_or_make_var: Callable[
                       [Any, bool, Optional[FxGraph], list[StorePos]],
                       Variable],
                   _fx_graph: Optional[FxGraph] = None,
                   extract_code_at_start: list[StorePos] = []) -> "ModuleVar":
        return cls(value, need_guard_check, extract_code_at_start)


class FunctionVar(Variable):

    def __init__(self,
                 func: Callable[..., Any],
                 need_guard_check: bool,
                 extract_code_at_start: list[StorePos] = []) -> None:
        super().__init__(need_guard_check, func, extract_code_at_start)

    def make_guard_inner(self, codegen: "GuardFnCodegen",
                         pos: StorePos) -> None:
        codegen.add_id_check(f"id({pos}) == {id(self.obj)}", self.obj)

    def make_output_inner(self, name_in_graph_fn: str, store_pos: StorePos,
                          codegen: "GraphFnCodegen", in_return: bool,
                          idx: int) -> None:
        name_in_codegen = codegen.add_var(self.obj, f"_{self.obj.__name__}")
        codegen.output(name_in_graph_fn, store_pos, name_in_codegen, in_return,
                       idx)

    @classmethod
    def from_value(cls,
                   value: Callable[..., Any],
                   need_guard_check: bool,
                   _get_or_make_var: Callable[
                       [Any, bool, Optional[FxGraph], list[StorePos]],
                       Variable],
                   _fx_graph: Optional[FxGraph] = None,
                   extract_code_at_start: list[StorePos] = []) -> "FunctionVar":
        return cls(value, need_guard_check, extract_code_at_start)


class RangeVar(Variable):
    start: Optional[int]
    stop: Optional[int]
    step: Optional[int]

    def __init__(self,
                 start: Optional[int],
                 stop: Optional[int],
                 step: Optional[int],
                 need_guard_check: bool,
                 obj: range,
                 extract_code_at_start: list[StorePos] = []) -> None:
        super().__init__(need_guard_check, obj, extract_code_at_start)
        self.start = start
        self.stop = stop
        self.step = step

    def make_guard_inner(self, codegen: "GuardFnCodegen",
                         pos: StorePos) -> None:
        codegen.add_check(
            f"{pos} == range({self.start}, {self.stop}, {self.step})")

    def make_output_inner(self, name_in_graph_fn: str, store_pos: StorePos,
                          codegen: "GraphFnCodegen", in_return: bool,
                          idx: int) -> None:
        codegen.output(name_in_graph_fn, store_pos,
                       f"range({self.start}, {self.stop}, {self.step})",
                       in_return, idx)

    @classmethod
    def from_value(cls,
                   value: range,
                   need_guard_check: bool,
                   _get_or_make_var: Callable[
                       [Any, bool, Optional[FxGraph], list[StorePos]],
                       Variable],
                   _fx_graph: Optional[FxGraph] = None,
                   extract_code_at_start: list[StorePos] = []) -> "RangeVar":
        return cls(value.start, value.stop, value.step, need_guard_check, value,
                   extract_code_at_start)
