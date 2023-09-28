from typing import Any, get_args, Optional, Tuple, Generic
from .variables.base import Variable
from .variables import CONST_TYPES, ScalarVar, make_var_from_value
from .variables.tuple_ import TupleVar
from .utils import NullObject, ReadOnlyObject
from .store_pos import StorePos
from .fx_graph import FxGraph


class ObjectTable:
    objs: dict[int, Variable]  # id -> object
    # Python caches small integers, so int variables don't have unique ids
    objs_no_id: list[Variable]
    read_only: 'ReadOnlyObjectTable'

    def __init__(self) -> None:
        self.objs = {}
        self.objs_no_id = []
        self.read_only = ReadOnlyObjectTable(self)

    def add(self, var: Variable, value: Any) -> None:
        if isinstance(value, bool):
            self.objs_no_id.append(var)
        elif id(value) in self.objs:
            old_var = self.objs[id(value)]
            old_var.extract_code_at_start.extend(var.extract_code_at_start)
            old_var.need_guard_check |= var.need_guard_check
        else:
            self.objs[id(value)] = var
            var.add_subvars_to_table(self)

    def add_by_id(self, var: Variable, idx: int) -> None:
        self.objs[idx] = var
        var.add_subvars_to_table(self)

    def get_all(self) -> list[Variable]:
        return list(self.objs.values()) + self.objs_no_id

    def get(self, value: Any, allow_unexist_const: bool = False) -> Variable:
        if isinstance(value, bool):
            return ScalarVar(value, False)
        elif id(value) in self.objs:
            return self.objs[id(value)]
        elif allow_unexist_const:
            if isinstance(value, get_args(CONST_TYPES)) or isinstance(
                    value, (list, tuple, set)):
                return make_var_from_value(value, False, self.read_only)
        raise RuntimeError(f"Object {value} not found in object table")

    def get_or_none(self, value: Any) -> Optional[Variable]:
        if id(value) in self.objs:
            return self.objs[id(value)]
        else:
            return None

    def get_or_make_var(self,
                        value: Any,
                        need_guard_check: bool,
                        fx_graph: Optional[FxGraph] = None,
                        extract_code_at_start: list[StorePos] = []) -> Variable:
        if isinstance(value, bool):
            return ScalarVar(value, need_guard_check, extract_code_at_start)
        elif id(value) in self.objs:
            return self.objs[id(value)]
        else:
            return make_var_from_value(value, need_guard_check, self.read_only,
                                       fx_graph, extract_code_at_start)

    def get_by_id(self, idx: int) -> Variable:
        return self.objs[idx]

    def contains(self, value: Any) -> bool:
        return id(value) in self.objs

    def contains_by_id(self, idx: int) -> bool:
        return idx in self.objs


class ReadOnlyObjectTable:
    table: ObjectTable

    def __init__(self, table: ObjectTable) -> None:
        self.table = table

    def get_all(self) -> list[Variable]:
        return self.table.get_all()

    def get(self, value: Any, allow_unexist_const: bool = False) -> Variable:
        return self.table.get(value, allow_unexist_const)

    def get_or_none(self, value: Any) -> Optional[Variable]:
        return self.table.get_or_none(value)

    def get_or_make_var(self,
                        value: Any,
                        need_guard_check: bool,
                        fx_graph: Optional[FxGraph] = None,
                        extract_code_at_start: list[StorePos] = []) -> Variable:
        return self.table.get_or_make_var(value, need_guard_check, fx_graph,
                                          extract_code_at_start)

    def get_by_id(self, idx: int) -> Variable:
        return self.table.get_by_id(idx)

    def contains(self, value: Any) -> bool:
        return self.table.contains(value)

    def contains_by_id(self, idx: int) -> bool:
        return self.table.contains_by_id(idx)