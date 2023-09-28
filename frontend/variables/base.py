from dataclasses import dataclass
from abc import abstractmethod
from typing import Any, TYPE_CHECKING, Optional, Tuple, Iterable, Callable
from copy import copy
from ..fx_graph import FxGraph
from ..store_pos import StorePos
if TYPE_CHECKING:
    import torch.fx
    from ..pycode_generator import GraphFnCodegen, GuardFnCodegen
    from ..fx_graph import FxGraph, NodeArgs
    from ..object_table import ObjectTable


@dataclass
class Variable:
    need_guard_check: bool
    extract_code_at_start: list[StorePos]
    obj: Any
    prev: Optional['Variable'] = None

    def __init__(self, need_guard_check: bool, obj: Any,
                 extract_code_at_start: list[StorePos]) -> None:
        self.need_guard_check = need_guard_check
        self.obj = obj
        self.extract_code_at_start = extract_code_at_start
        if need_guard_check:
            assert len(extract_code_at_start) > 0

    @classmethod
    @abstractmethod
    def from_value(self,
                   value: Any,
                   need_guard_check: bool,
                   get_or_make_var: Callable[
                       [Any, bool, Optional[FxGraph], list[StorePos]],
                       'Variable'],
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

    def make_output(self, name_in_graph_fn: str, store_pos: StorePos,
                    codegen: "GraphFnCodegen", in_return: bool,
                    idx: int) -> None:
        if idx in codegen.id2name:
            codegen.output(name_in_graph_fn, store_pos, codegen.id2name[idx],
                           in_return, 0)
        else:
            self.make_output_inner(name_in_graph_fn, store_pos, codegen,
                                   in_return, idx)

    @abstractmethod
    def make_output_inner(self, name_in_graph_fn: str, store_pos: StorePos,
                          codegen: "GraphFnCodegen", in_return: bool,
                          idx: int) -> None:
        raise NotImplementedError

    @abstractmethod
    def as_fx_node(self) -> "NodeArgs":
        raise NotImplementedError

    def add_subvars_to_table(self, table: 'ObjectTable') -> None:
        pass

    def set_prev(self, prev: Optional['Variable']) -> None:
        self.prev = prev

    def get_subvars_with_idx(self) -> Iterable[Tuple["Variable", int]]:
        return []

    def get_oldest_var(self) -> "Variable":
        ret = self
        while ret.prev is not None:
            ret = ret.prev
        return ret
