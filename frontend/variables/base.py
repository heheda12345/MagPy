from dataclasses import dataclass
from abc import abstractmethod
from typing import Any, TYPE_CHECKING, Optional
from ..fx_graph import FxGraph
from ..store_pos import StorePos
if TYPE_CHECKING:
    import torch.fx
    from ..pycode_generator import GraphFnCodegen, GuardFnCodegen
    from ..fx_graph import FxGraph, NodeArgs
    from ..object_table import ReadOnlyObjectTable, ObjectTable


@dataclass
class Variable:
    need_guard_check: bool
    extract_code_at_start: list[StorePos]
    prev: Optional['Variable'] = None

    def __init__(self,
                 need_guard_check: bool,
                 extract_code_at_start: list[StorePos] = []) -> None:
        self.need_guard_check = need_guard_check
        self.extract_code_at_start = extract_code_at_start
        if need_guard_check:
            assert len(extract_code_at_start) > 0

    @classmethod
    @abstractmethod
    def from_value(self,
                   value: Any,
                   need_guard_check: bool,
                   object_table: 'ReadOnlyObjectTable',
                   fx_graph: Optional[FxGraph] = None,
                   extract_code_at_start: list[StorePos] = []) -> 'Variable':
        raise NotImplementedError

    def make_guard(self, codegen: "GuardFnCodegen") -> None:
        if self.need_guard_check:
            assert len(self.extract_code_at_start) > 0
            for pos in self.extract_code_at_start:
                self.make_guard_inner(codegen, pos)

    @abstractmethod
    def make_guard_inner(self, codegen: "GuardFnCodegen",
                         pos: StorePos) -> None:
        raise NotImplementedError

    @abstractmethod
    def make_output(self, name_in_graph_fn: str, store_pos: StorePos,
                    codegen: "GraphFnCodegen") -> None:
        raise NotImplementedError

    @abstractmethod
    def make_temp(self, name_in_graph_fn: str, store_pos: StorePos,
                  codegen: "GraphFnCodegen") -> None:
        raise NotImplementedError

    @abstractmethod
    def as_fx_node(self) -> "NodeArgs":
        raise NotImplementedError

    def add_subvars_to_table(self, table: 'ObjectTable') -> None:
        pass
