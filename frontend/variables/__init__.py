from typing import Any
from .base import Guard
from .tensor import TensorVar
from .scalar import ScalarGuard
from .result_writer import ResultWriter

ty2guard = {
    float: ScalarGuard,
    int: ScalarGuard,
}


def make_guard(extract_code: str, value: Any) -> Guard:
    if type(value) in ty2guard:
        return ty2guard[type(value)](extract_code, value)
    else:
        raise NotImplementedError(f"unknown type: {type(value)}")


__all__ = [
    'Guard',
    'make_guard',
    'ResultWriter',
]