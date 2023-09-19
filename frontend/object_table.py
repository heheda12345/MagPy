from typing import Any, get_args, Optional, Tuple
from .variables.base import Variable
from .variables import CONST_TYPES, ScalarVar, make_var_from_value
from .variables.tuple_ import TupleVar
from .utils import NullObject


class ObjectTable:
    objs: dict[int, Variable]  # id -> object
    # Python caches small integers, so int variables don't have unique ids
    objs_no_id: list[Variable]

    def __init__(self) -> None:
        self.objs = {}
        self.objs_no_id = []

    def add(self, var: Variable, value: Any) -> None:
        if isinstance(value, int) and value >= -5 and value <= 256:
            self.objs_no_id.append(var)
        else:
            self.objs[id(value)] = var

    def get_all(self) -> list[Variable]:
        return list(self.objs.values()) + self.objs_no_id

    def get(self, value: Any, allow_unexist_const: bool = False) -> Variable:
        if isinstance(value, int) and value >= -5 and value <= 256:
            return ScalarVar(value, False)
        elif id(value) in self.objs:
            return self.objs[id(value)]
        elif allow_unexist_const and isinstance(value, get_args(CONST_TYPES)):
            return make_var_from_value(value, False)
        elif isinstance(value, tuple):
            return TupleVar(value, False)
        raise RuntimeError(f"Object {value} not found in object table")

    def get_or_none(self, value: Any) -> Optional[Variable]:
        if id(value) in self.objs:
            return self.objs[id(value)]
        else:
            return None

    def contains(self, value: Any) -> bool:
        return id(value) in self.objs
