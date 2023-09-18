from typing import TYPE_CHECKING, Union, Optional

import torch.fx

from frontend.pycode_generator import GuardFnCodegen
from .base import Variable
from ..fx_graph import FxGraph
from ..store_pos import StorePos, unknown_pos
if TYPE_CHECKING:
    from ..pycode_generator import GraphFnCodegen, GuardFnCodegen

ScalarType = Union[int, float, bool, str]


class TorchModuleVar(Variable):
    module: torch.nn.Module

    def __init__(self,
                 value: torch.nn.Module,
                 need_guard_check: bool,
                 extract_code_at_start: StorePos = unknown_pos) -> None:
        super().__init__(need_guard_check, extract_code_at_start)
        assert extract_code_at_start != unknown_pos
        self.module = value

    @classmethod
    def from_value(
            cls,
            value: torch.nn.Module,
            need_guard_check: bool,
            _fx_graph: Optional[FxGraph] = None,
            extract_code_at_start: StorePos = unknown_pos) -> "TorchModuleVar":
        return cls(value, need_guard_check, extract_code_at_start)

    def make_guard_inner(self, codegen: GuardFnCodegen) -> None:
        codegen.add_check(
            f"id({self.extract_code_at_start}) == {id(self.module)}")

    def make_output(self, name_in_graph_fn: str, store_pos: StorePos,
                    codegen: "GraphFnCodegen") -> None:
        codegen.output(name_in_graph_fn, store_pos,
                       str(self.extract_code_at_start))
