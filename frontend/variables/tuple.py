from typing import TYPE_CHECKING, Optional, Tuple, Any
from .base import Variable
from ..fx_graph import NodeArgs, FxGraph
from ..store_pos import StorePos
import torch
if TYPE_CHECKING:
    from ..pycode_generator import GraphFnCodegen, GuardFnCodegen


class TupleVar(Variable):

    def __init__(self,
                 value: tuple[Any, ...],
                 need_guard_check: bool,
                 extract_code_at_start: list[StorePos] = []) -> None:
        super().__init__(need_guard_check, extract_code_at_start)
        self.value = value

    def make_guard_inner(self, codegen: "GuardFnCodegen",
                         pos: StorePos) -> None:
        codegen.add_check(f"{pos} == {self.value}")

    def make_output(self, name_in_graph_fn: str, store_pos: StorePos,
                    codegen: "GraphFnCodegen") -> None:
        # print(f"come to make output:{self.value}")
        # for sub_value in self.value:
        #     if isinstance(sub_value, torch.Tensor):
        #         id(sub_value)
        codegen.output(name_in_graph_fn, store_pos, str(self.value))

    @classmethod
    def from_value(cls,
                   value: Tuple[Any, ...],
                   need_guard_check: bool,
                   _fx_graph: Optional[FxGraph] = None,
                   extract_code_at_start: list[StorePos] = []) -> "TupleVar":
        return cls(value, need_guard_check, extract_code_at_start)

    def as_fx_node(self) -> NodeArgs:
        return self.value