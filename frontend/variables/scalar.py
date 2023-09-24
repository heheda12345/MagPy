from typing import TYPE_CHECKING, Union, Optional

import torch.fx
from .base import Variable
from ..pycode_writer import get_float_string
from ..fx_graph import NodeArgs, FxGraph
from ..store_pos import StorePos
if TYPE_CHECKING:
    from ..pycode_generator import GraphFnCodegen, GuardFnCodegen

ScalarType = Union[int, float, bool, str]


class ScalarVar(Variable):
    value: ScalarType

    def __init__(self,
                 value: ScalarType,
                 need_guard_check: bool,
                 extract_code_at_start: list[StorePos] = []) -> None:
        super().__init__(need_guard_check, extract_code_at_start)
        self.value = value

    def make_guard_inner(self, codegen: "GuardFnCodegen",
                         pos: StorePos) -> None:
        if type(self.value) == float:
            codegen.add_check(f"{pos} == {get_float_string(self.value)}")
            codegen.add_import("struct")
        else:
            codegen.add_check(f"{pos} == {self.value}")

    def make_output(self, name_in_graph_fn: str, store_pos: StorePos,
                    codegen: "GraphFnCodegen") -> None:
        if type(self.value) == float:
            codegen.output(name_in_graph_fn, store_pos,
                           f"{get_float_string(self.value)} # {self.value}")
            codegen.add_import("struct")
        else:
            codegen.output(name_in_graph_fn, store_pos, str(self.value))

    def make_temp(self, name_in_graph_fn: str, store_pos: StorePos,
                  codegen: "GraphFnCodegen") -> None:
        codegen.add_temp(name_in_graph_fn, store_pos, str(self.value))

    @classmethod
    def from_value(cls,
                   value: ScalarType,
                   need_guard_check: bool,
                   _fx_graph: Optional[FxGraph] = None,
                   extract_code_at_start: list[StorePos] = []) -> "ScalarVar":
        return cls(value, need_guard_check, extract_code_at_start)

    def as_fx_node(self) -> NodeArgs:
        return self.value
