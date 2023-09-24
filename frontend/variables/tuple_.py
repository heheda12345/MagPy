from typing import TYPE_CHECKING, Optional, Tuple, Any
from .base import Variable
from ..fx_graph import NodeArgs, FxGraph
from ..store_pos import StorePos
import torch
if TYPE_CHECKING:
    from ..pycode_generator import GraphFnCodegen, GuardFnCodegen


class TupleVar(Variable):
    objs: list[Variable]
    length: int

    def __init__(self,
                 value: tuple[Any, ...],
                 need_guard_check: bool,
                 extract_code_at_start: list[StorePos] = []) -> None:
        super().__init__(need_guard_check, extract_code_at_start)
        self.value = value
        self.length = len(value)
        self.objs = []

    def make_guard_inner(self, codegen: "GuardFnCodegen", pos: StorePos) -> None:
        codegen.add_check(f"111==111 and len({pos}) == {self.length} and 222==222")

    def make_output(self, name_in_graph_fn: str, store_pos: StorePos,
                    codegen: "GraphFnCodegen") -> None:
        for j, var in enumerate(self.objs):
            var.make_temp(f"{name_in_graph_fn}_{j}", store_pos, codegen)

        codegen.output(
            name_in_graph_fn, store_pos,
            f"({','.join(f'{name_in_graph_fn}_{j}' for j in range(len(self.objs)))},)"
        )

    def make_temp(self, name_in_graph_fn: str, store_pos: StorePos,
                  codegen: "GraphFnCodegen") -> None:
        for j, var in enumerate(self.objs):
            var.make_temp(f"{name_in_graph_fn}_{j}", store_pos, codegen)
        codegen.add_temp(
            name_in_graph_fn, store_pos,
            f"({','.join(f'{name_in_graph_fn}_{j}' for j in range(len(self.objs)))})"
        )

    @classmethod
    def from_value(cls,
                   value: Tuple[Any, ...],
                   need_guard_check: bool,
                   _fx_graph: Optional[FxGraph] = None,
                   extract_code_at_start: list[StorePos] = []) -> "TupleVar":
        return cls(value, need_guard_check, extract_code_at_start)

    def as_fx_node(self) -> NodeArgs:
        return self.value