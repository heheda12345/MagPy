from typing import TYPE_CHECKING, Union, Optional, Callable, Any
from frontend.pycode_generator import GraphFnCodegen

import torch.fx
from types import ModuleType
from enum import Enum
from .base import Variable
from ..fx_graph import NodeArgs, FxGraph
from ..store_pos import StorePos
if TYPE_CHECKING:
    from ..pycode_generator import GraphFnCodegen, GuardFnCodegen


class AnyVar(Variable):

    def __init__(self, need_guard_check: bool, obj: Any,
                 extract_code_at_start: list[StorePos]) -> None:
        super().__init__(need_guard_check, obj, extract_code_at_start)

    def make_guard_inner(self, codegen: "GuardFnCodegen",
                         pos: StorePos) -> None:
        pass

    def make_output_inner(self, name_in_graph_fn: str, store_pos: StorePos,
                          codegen: "GraphFnCodegen", in_return: bool,
                          idx: int) -> None:
        extract_pos = self.fetch_extract_code_at_start()
        assert len(extract_pos) > 0
        codegen.output(name_in_graph_fn, store_pos, str(extract_pos[0]),
                       in_return, idx)

    @classmethod
    def from_value(cls, value: None, need_guard_check: bool,
                   _get_or_make_var: Callable[
                       [Any, bool, Optional[FxGraph], list[StorePos]],
                       Variable], _fx_graph: Optional[FxGraph],
                   extract_code_at_start: list[StorePos]) -> "AnyVar":
        return cls(need_guard_check, value, extract_code_at_start)

    def as_fx_node(self) -> NodeArgs:
        raise NotImplementedError(self.obj)
