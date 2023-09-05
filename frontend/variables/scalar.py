from typing import Any, TYPE_CHECKING

import torch.fx
from .base import Variable
from ..pycode_writer import get_float_string
if TYPE_CHECKING:
    from ..pycode_generator import GraphFnCodegen, GuardFnCodegen
    from ..fx_graph import FxGraph


class ScalarVar(Variable):
    value: Any

    def __init__(self,
                 value: Any,
                 need_guard_check: bool,
                 extract_code_at_start: str = "") -> None:
        super().__init__(need_guard_check, extract_code_at_start)
        self.value = value

    def make_guard_inner(self, codegen: "GuardFnCodegen") -> None:
        if type(self.value) == float:
            codegen.add_check(
                f"{self.extract_code_at_start} == {get_float_string(self.value)}"
            )
            codegen.add_import("struct")
        else:
            codegen.add_check(f"{self.extract_code_at_start} == {self.value}")

    def make_output(self, target_name: str, codegen: "GraphFnCodegen") -> None:
        if type(self.value) == float:
            codegen.output(target_name,
                           f"{get_float_string(self.value)} # {self.value}")
            codegen.add_import("struct")
        else:
            codegen.output(target_name, str(self.value))

    @classmethod
    def from_value(cls,
                   value: Any,
                   need_guard_check: bool,
                   _fx_graph: "torch.fx.Graph",
                   extract_code_at_start: str = "") -> "ScalarVar":
        return cls(value, need_guard_check, extract_code_at_start)

    def as_proxy(self) -> torch.fx.Proxy:
        raise NotImplementedError
