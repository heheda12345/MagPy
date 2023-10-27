from typing import Any, get_args, Optional, Tuple, Generic, Callable
from types import CodeType
from .variables.base import Variable, HelperFunctions
from .variables.any_ import AnyVar
from .variables import CONST_TYPES, ScalarVar, make_var_from_value
from .variables.tuple_ import TupleVar
from .utils import NullObject, ReadOnlyObject
from .store_pos import StorePos
from .fx_graph import FxGraph
import torch


class ObjectTable:
    objs: dict[int, Variable]  # id -> object
    # Python caches small integers, so int variables don't have unique ids
    objs_no_id: list[Variable]
    helper_functions: HelperFunctions

    def __init__(self, gen_by_caller: Callable[[Any], bool],
                 mark_cannot_guard: Callable[[], None]) -> None:
        self.objs = {}
        self.objs_no_id = []
        self.helper_functions = HelperFunctions(self.get_or_make_var,
                                                gen_by_caller,
                                                mark_cannot_guard)

    def add(self, var: Variable, value: Any) -> None:
        if id(value) in self.objs:
            old_var = self.objs[id(value)]
            if isinstance(old_var, AnyVar) and not isinstance(var, AnyVar):
                self.objs[id(value)] = var
                var, old_var = old_var, var
            for pos in var.extract_code_at_start:
                old_var.add_extract_code_at_start(pos)
            old_var.need_guard_check |= var.need_guard_check
        else:
            self.add_by_id(var, id(value))
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
        if old_var is not None:
            for attr_name, attr_var in old_var.modified_attrs.items():
                if attr_name not in var.modified_attrs:
                    var.add_modified_attr(attr_name, attr_var)

    def get_all(self) -> list[Variable]:
        return list(self.objs.values()) + self.objs_no_id

    def get_all_with_id(self) -> list[Tuple[int, Variable]]:
        return list(self.objs.items())

    def get(self, value: Any, allow_unexist_const: bool = False) -> Variable:
        if id(value) in self.objs:
            return self.objs[id(value)]
        elif allow_unexist_const:
            if isinstance(value, get_args(CONST_TYPES)) or isinstance(
                    value, (list, tuple, set, dict, CodeType)):
                return make_var_from_value(value, False, self.helper_functions)
        raise RuntimeError(
            f"Object {value}({id(value)}) not found in object table")

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
        if id(value) in self.objs:
            return self.objs[id(value)]
        else:
            return make_var_from_value(value, need_guard_check,
                                       self.helper_functions, fx_graph,
                                       extract_code_at_start)

    def get_by_id(self, idx: int) -> Variable:
        return self.objs[idx]

    def contains(self, value: Any) -> bool:
        return id(value) in self.objs

    def contains_by_id(self, idx: int) -> bool:
        return idx in self.objs
