from dataclasses import dataclass
from abc import abstractmethod
from typing import Any, TYPE_CHECKING, Optional, Tuple, Iterable, Callable
from copy import copy

from frontend.utils import add_force_graph_break

from ..c_api import get_miss_locals
from ..fx_graph import FxGraph
from ..store_pos import StorePos, StoreInAttr

if TYPE_CHECKING:
    import torch.fx
    from ..pycode_generator import GraphFnCodegen, GuardFnCodegen
    from ..fx_graph import FxGraph, NodeArgs
    from ..object_table import ObjectTable


@dataclass
class HelperFunctions:
    get_or_make_var: Callable[[Any, bool, Optional[FxGraph], list[StorePos]],
                              'Variable']
    gen_by_caller: Callable[[Any], bool]
    mark_cannot_guard: Callable[[], None]


@dataclass
class Variable:
    need_guard_check: bool
    extract_code_at_start: list[StorePos]
    extract_code_hashs: set[int]
    obj: Any
    modified_attrs: dict[str, 'Variable']
    prev: Optional['Variable'] = None
    succ: Optional['Variable'] = None

    def __init__(self, need_guard_check: bool, obj: Any,
                 extract_code_at_start: list[StorePos]) -> None:
        from ..guard_tracker import trackers
        for i in get_miss_locals(trackers[-1].frame_id):
            for j in extract_code_at_start:
                if (i == f"{j}"):
                    print(i)
                    print("--------warning--------")

        self.need_guard_check = need_guard_check
        self.obj = obj
        self.extract_code_at_start = extract_code_at_start
        self.extract_code_hashs = set()
        for pos in extract_code_at_start:
            self.extract_code_hashs.add(str(pos).__hash__())
        if need_guard_check:
            assert len(extract_code_at_start) > 0
        self.modified_attrs = dict()

    @classmethod
    @abstractmethod
    def from_value(
        self,
        value: Any,
        need_guard_check: bool,
        _helper_functions: 'HelperFunctions',
        fx_graph: Optional[FxGraph],
        extract_code_at_start: list[StorePos],
    ) -> 'Variable':
        raise NotImplementedError

    def make_guard(self, codegen: "GuardFnCodegen") -> None:
        if self.need_guard_check:
            assert len(self.extract_code_at_start) > 0
            for pos in self.extract_code_at_start:
                pos.add_name_to_fn(codegen)
                self.make_guard_inner(codegen, pos)

    @abstractmethod
    def make_guard_inner(self, codegen: "GuardFnCodegen",
                         pos: StorePos) -> None:
        raise NotImplementedError

    def make_output(self, name_in_graph_fn: str, store_pos: StorePos,
                    codegen: "GraphFnCodegen", in_return: bool,
                    idx: int) -> None:
        if self.succ is not None:
            return self.succ.make_output(name_in_graph_fn, store_pos, codegen,
                                         in_return, idx)
        if idx in codegen.id2name:
            codegen.output(name_in_graph_fn, store_pos, codegen.id2name[idx],
                           in_return, 0)
        else:
            self.make_output_inner(name_in_graph_fn, store_pos, codegen,
                                   in_return, idx)
            for attr, var in self.modified_attrs.items():
                var.make_output(f'{name_in_graph_fn}_dot_{attr}',
                                StoreInAttr(store_pos, id(self.obj), attr),
                                codegen, False, id(getattr(self.obj, attr)))
                codegen.add_stmt(
                    f"setattr({name_in_graph_fn}, '{attr}', {name_in_graph_fn}_dot_{attr})"
                )

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
        if prev is not None:
            prev.succ = self

    def get_subvars_with_idx(self) -> Iterable[Tuple["Variable", int]]:
        return []

    def get_oldest_var(self) -> "Variable":
        ret = self
        while ret.prev is not None:
            ret = ret.prev
        return ret

    def disable_guard_check(self) -> None:
        self.need_guard_check = False

    def clear_extract_code_at_start(self) -> None:
        self.extract_code_at_start = []
        self.extract_code_hashs = set()

    def add_extract_code_at_start(self, pos: StorePos) -> None:
        from ..guard_tracker import trackers
        for i in get_miss_locals(trackers[-1].frame_id):
            if i == f"{pos}":
                print(i)
                print("--------warning--------")

        hash_value = str(pos).__hash__()
        if hash_value not in self.extract_code_hashs:
            self.extract_code_at_start.append(pos)
            self.extract_code_hashs.add(hash_value)

    def add_modified_attr(self, attr: str, var: 'Variable') -> None:
        self.modified_attrs[attr] = var

    def fetch_extract_code_at_start(self) -> list[StorePos]:

        def is_same(a: dict[str, Variable], b: dict[str, Variable]) -> bool:
            if len(a) != len(b):
                return False
            for k, v in a.items():
                if k not in b or id(b[k]) != id(v):
                    return False
            return True

        prev = self
        while prev.prev is not None and not is_same(prev.prev.modified_attrs,
                                                    prev.modified_attrs):
            prev = prev.prev
        return prev.extract_code_at_start
