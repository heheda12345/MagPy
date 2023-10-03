from typing import Any


class Dynamic:
    pass


class ScalarWithUnknownValue(Dynamic):
    pass


dynamics = {}
dynamic_refs = {}


def mark_dynamic(obj: Any, dyn: Dynamic) -> None:
    idx = id(obj)
    dynamics[idx] = dyn
    dynamic_refs[idx] = obj


def contains(obj: Any) -> bool:
    idx = id(obj)
    return idx in dynamics


def contains_by_id(idx: int) -> bool:
    return idx in dynamics