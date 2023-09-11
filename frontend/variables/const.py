from typing import TYPE_CHECKING, Union, Optional

import torch.fx
from .base import Variable
from ..pycode_writer import get_float_string
from ..fx_graph import ProxyArgs, FxGraph
from ..cache import StorePos
from ..utils import NullObject, null_object
if TYPE_CHECKING:
    from ..pycode_generator import GraphFnCodegen, GuardFnCodegen


class NoneVar(Variable):

    def __init__(self,
                 need_guard_check: bool,
                 extract_code_at_start: str = "") -> None:
        super().__init__(need_guard_check, extract_code_at_start)

    def make_guard_inner(self, codegen: "GuardFnCodegen") -> None:
        codegen.add_check(f"{self.extract_code_at_start} is None")

    def make_output(self, name_in_graph_fn: str, store_pos: StorePos,
                    codegen: "GraphFnCodegen") -> None:
        codegen.output(name_in_graph_fn, store_pos, "None")

    @classmethod
    def from_value(cls,
                   value: None,
                   need_guard_check: bool,
                   _fx_graph: Optional[FxGraph] = None,
                   extract_code_at_start: str = "") -> "NoneVar":
        return cls(need_guard_check, extract_code_at_start)

    def as_proxy(self) -> ProxyArgs:
        return None


class NullVar(Variable):

    def __init__(self,
                 need_guard_check: bool,
                 extract_code_at_start: str = "") -> None:
        super().__init__(need_guard_check, extract_code_at_start)

    def make_guard_inner(self, codegen: "GuardFnCodegen") -> None:
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
                   extract_code_at_start: str = "") -> "NullVar":
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
                 extract_code_at_start: str = "") -> None:
        super().__init__(need_guard_check, extract_code_at_start)
        self.start = start
        self.stop = stop
        self.step = step

    def make_guard_inner(self, codegen: "GuardFnCodegen") -> None:
        codegen.add_check(
            f"{self.extract_code_at_start} == slice({self.start}, {self.stop}, {self.step})"
        )

    def make_output(self, name_in_graph_fn: str, store_pos: StorePos,
                    codegen: "GraphFnCodegen") -> None:
        codegen.output(name_in_graph_fn, store_pos, "None")

    @classmethod
    def from_value(cls,
                   value: slice,
                   need_guard_check: bool,
                   _fx_graph: Optional[FxGraph] = None,
                   extract_code_at_start: str = "") -> "SliceVar":
        return cls(value.start, value.stop, value.step, need_guard_check,
                   extract_code_at_start)

    def as_proxy(self) -> ProxyArgs:
        return slice(self.start, self.stop, self.step)