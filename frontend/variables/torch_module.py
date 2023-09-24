from typing import TYPE_CHECKING, Union, Optional
import torch.fx
from frontend.pycode_generator import GraphFnCodegen, GuardFnCodegen
from .base import Variable
from ..fx_graph import FxGraph
from ..store_pos import StorePos
if TYPE_CHECKING:
    from ..pycode_generator import GraphFnCodegen, GuardFnCodegen

ScalarType = Union[int, float, bool, str]


class TorchModuleVar(Variable):
    module: torch.nn.Module

    def __init__(self,
                 value: torch.nn.Module,
                 need_guard_check: bool,
                 extract_code_at_start: list[StorePos] = []) -> None:
        super().__init__(need_guard_check, extract_code_at_start)
        assert len(extract_code_at_start) > 0
        self.module = value

    @classmethod
    def from_value(
            cls,
            value: torch.nn.Module,
            need_guard_check: bool,
            _fx_graph: Optional[FxGraph] = None,
            extract_code_at_start: list[StorePos] = []) -> "TorchModuleVar":
        return cls(value, need_guard_check, extract_code_at_start)

    def make_guard_inner(self, codegen: GuardFnCodegen, pos: StorePos) -> None:
        codegen.add_id_check(f"id({pos}) == {id(self.module)}", self.module)

    def make_output(self, name_in_graph_fn: str, store_pos: StorePos,
                    codegen: "GraphFnCodegen") -> None:
        assert len(self.extract_code_at_start) > 0
        codegen.output(name_in_graph_fn, store_pos,
                       str(self.extract_code_at_start[0]))

    def make_temp(self, name_in_graph_fn: str, store_pos: StorePos,
                  codegen: GraphFnCodegen) -> None:
        return super().make_temp(name_in_graph_fn, store_pos, codegen)