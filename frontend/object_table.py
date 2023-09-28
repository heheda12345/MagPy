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

    def __init__(self) -> None:
        self.objs = {}
        self.objs_no_id = []

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
        assert idx not in self.objs
        self.objs[idx] = var
        var.add_subvars_to_table(self)

    def update_by_id(self, var: Variable, idx: int) -> None:
        if self.contains_by_id(idx):
            old_var = self.objs[idx]
        else:
            old_var = None
        var.set_prev(old_var)
        self.objs[idx] = var

    def get_all(self) -> list[Variable]:
        return list(self.objs.values()) + self.objs_no_id

    def get_all_with_id(self) -> list[Tuple[int, Variable]]:
        return list(self.objs.items())

    def get(self, value: Any, allow_unexist_const: bool = False) -> Variable:
        if isinstance(value, bool):
            return ScalarVar(value, False)
        elif id(value) in self.objs:
            return self.objs[id(value)]
        elif allow_unexist_const:
            if isinstance(value, get_args(CONST_TYPES)) or isinstance(
                    value, (list, tuple, set)):
                return make_var_from_value(value, False, self.get_or_make_var)
        raise RuntimeError(f"Object {value} not found in object table")

    def get_or_none(self, value: Any) -> Optional[Variable]:
        if id(value) in self.objs:
            return self.objs[id(value)]
        else:
            return None

    def get_or_none_by_id(self, idx: int) -> Optional[Variable]:
        if idx in self.objs:
            return self.objs[idx]
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
            return make_var_from_value(value, need_guard_check,
                                       self.get_or_make_var, fx_graph,
                                       extract_code_at_start)

    def get_by_id(self, idx: int) -> Variable:
        return self.objs[idx]

    def contains(self, value: Any) -> bool:
        return id(value) in self.objs

    def contains_by_id(self, idx: int) -> bool:
        return idx in self.objs
