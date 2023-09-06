from typing import Any, get_args
from .variables.base import Variable
from .variables.scalar import ScalarVar, ScalarType


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
        elif allow_unexist_const and isinstance(value, get_args(ScalarType)):
            return ScalarVar(value, False)
        else:
            raise RuntimeError(f"Object {value} not found in object table")

    def contains(self, value: Any) -> bool:
        return id(value) in self.objs
