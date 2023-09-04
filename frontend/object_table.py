from typing import Any
from .variables.base import Variable


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
