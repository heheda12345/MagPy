from typing import TYPE_CHECKING, Union, Optional, Callable, Any

import torch.fx
from types import ModuleType
from enum import Enum
from .base import Variable
from ..pycode_writer import get_float_string
from ..fx_graph import ProxyArgs, FxGraph
from ..utils import NullObject, null_object
from ..store_pos import StorePos
if TYPE_CHECKING:
    from ..pycode_generator import GraphFnCodegen, GuardFnCodegen


class NoneVar(Variable):

    def __init__(self,
                 need_guard_check: bool,
                 extract_code_at_start: list[StorePos] = []) -> None:
        super().__init__(need_guard_check, extract_code_at_start)

    def make_guard_inner(self, codegen: "GuardFnCodegen",
                         pos: StorePos) -> None:
        for pos in self.extract_code_at_start:
            codegen.add_check(f"{pos} is None")

    def make_output(self, name_in_graph_fn: str, store_pos: StorePos,
                    codegen: "GraphFnCodegen") -> None:
        codegen.output(name_in_graph_fn, store_pos, "None")

    @classmethod
    def from_value(cls,
                   value: None,
                   need_guard_check: bool,
                   _fx_graph: Optional[FxGraph] = None,
                   extract_code_at_start: list[StorePos] = []) -> "NoneVar":
        return cls(need_guard_check, extract_code_at_start)

    def as_proxy(self) -> ProxyArgs:
        return None


class NullVar(Variable):

    def __init__(self,
                 need_guard_check: bool,
                 extract_code_at_start: list[StorePos] = []) -> None:
        super().__init__(need_guard_check, extract_code_at_start)

    def make_guard_inner(self, codegen: "GuardFnCodegen",
                         pos: StorePos) -> None:
        pass

    def make_output(self, name_in_graph_fn: str, store_pos: StorePos,
                    codegen: "GraphFnCodegen") -> None:
        name_in_codegen = codegen.add_var(null_object, "NULL_VAR")
        codegen.output(name_in_graph_fn, store_pos, f"{name_in_codegen} # NULL")

    @classmethod
    def from_value(cls,
                   value: NullObject,
                   need_guard_check: bool,
                   _fx_graph: Optional[FxGraph] = None,
                   extract_code_at_start: list[StorePos] = []) -> "NullVar":
        return cls(need_guard_check, extract_code_at_start)

    def as_proxy(self) -> ProxyArgs:
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
                 extract_code_at_start: list[StorePos] = []) -> None:
        super().__init__(need_guard_check, extract_code_at_start)
        self.start = start
        self.stop = stop
        self.step = step

    def make_guard_inner(self, codegen: "GuardFnCodegen",
                         pos: StorePos) -> None:
        codegen.add_check(
            f"{pos} == slice({self.start}, {self.stop}, {self.step})")

    def make_output(self, name_in_graph_fn: str, store_pos: StorePos,
                    codegen: "GraphFnCodegen") -> None:
        codegen.output(name_in_graph_fn, store_pos, "None")

    @classmethod
    def from_value(cls,
                   value: slice,
                   need_guard_check: bool,
                   _fx_graph: Optional[FxGraph] = None,
                   extract_code_at_start: list[StorePos] = []) -> "SliceVar":
        return cls(value.start, value.stop, value.step, need_guard_check,
                   extract_code_at_start)

    def as_proxy(self) -> ProxyArgs:
        return slice(self.start, self.stop, self.step)


class ObjectSrc(Enum):
    USER_DEFINED = 0
    TORCH = 1


torch_modules = set([torch])


class ModuleVar(Variable):
    module: ModuleType
    src: ObjectSrc

    def __init__(self,
                 module: ModuleType,
                 src: ObjectSrc,
                 need_guard_check: bool,
                 extract_code_at_start: list[StorePos] = []) -> None:
        super().__init__(need_guard_check, extract_code_at_start)
        self.module = module
        self.src = src

    def make_guard_inner(self, codegen: "GuardFnCodegen",
                         pos: StorePos) -> None:
        codegen.add_id_check(f"id({pos}) == {id(self.module)}", self.module)

    def make_output(self, name_in_graph_fn: str, store_pos: StorePos,
                    codegen: "GraphFnCodegen") -> None:
        name_in_codegen = codegen.add_var(self.module,
                                          f"MODULE_{self.module.__name__}")
        codegen.output(name_in_graph_fn, store_pos, name_in_codegen)

    @classmethod
    def from_value(cls,
                   value: ModuleType,
                   need_guard_check: bool,
                   _fx_graph: Optional[FxGraph] = None,
                   extract_code_at_start: list[StorePos] = []) -> "ModuleVar":
        if value in torch_modules:
            src = ObjectSrc.TORCH
        else:
            src = ObjectSrc.USER_DEFINED
        return cls(value, src, need_guard_check, extract_code_at_start)


class FunctionVar(Variable):
    func: Callable[..., Any]
    src: ObjectSrc

    def __init__(self,
                 func: Callable[..., Any],
                 src: ObjectSrc,
                 need_guard_check: bool,
                 extract_code_at_start: list[StorePos] = []) -> None:
        super().__init__(need_guard_check, extract_code_at_start)
        self.func = func
        self.src = src

    def make_guard_inner(self, codegen: "GuardFnCodegen",
                         pos: StorePos) -> None:
        codegen.add_id_check(f"id({pos}) == {id(self.func)}", self.func)

    def make_output(self, name_in_graph_fn: str, store_pos: StorePos,
                    codegen: "GraphFnCodegen") -> None:
        name_in_codegen = codegen.add_var(self.func, f"_{self.func.__name__}")
        codegen.output(name_in_graph_fn, store_pos, name_in_codegen)

    @classmethod
    def from_value(cls,
                   value: Callable[..., Any],
                   need_guard_check: bool,
                   _fx_graph: Optional[FxGraph] = None,
                   extract_code_at_start: list[StorePos] = []) -> "FunctionVar":
        return cls(value, ObjectSrc.USER_DEFINED, need_guard_check,
                   extract_code_at_start)
