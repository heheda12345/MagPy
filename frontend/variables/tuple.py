from typing import TYPE_CHECKING, Optional, Tuple, Any
from .base import Variable
from ..fx_graph import ProxyArgs, FxGraph
from ..cache import StorePos
if TYPE_CHECKING:
    from ..pycode_generator import GraphFnCodegen, GuardFnCodegen


class TupleVar(Variable):

    def __init__(self,
                 value: tuple[Any, ...],
                 need_guard_check: bool,
                 extract_code_at_start: str = "") -> None:
        super().__init__(need_guard_check, extract_code_at_start)
        self.value = value

    def make_guard_inner(self, codegen: "GuardFnCodegen") -> None:
        codegen.add_check(f"{self.extract_code_at_start} == {self.value}")

    def make_output(self, name_in_graph_fn: str, store_pos: StorePos,
                    codegen: "GraphFnCodegen") -> None:
        codegen.output(name_in_graph_fn, store_pos, str(self.value))

    @classmethod
    def from_value(cls,
                   value: Tuple[Any, ...],
                   need_guard_check: bool,
                   _fx_graph: Optional[FxGraph] = None,
                   extract_code_at_start: str = "") -> "TupleVar":
        return cls(value, need_guard_check, extract_code_at_start)

    def as_proxy(self) -> ProxyArgs:
        return self.value