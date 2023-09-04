from typing import Any
import torch
from .base import Variable
# from .tensor import TensorVar
from .scalar import ScalarVar

ty2var = {
    float: ScalarVar,
    int: ScalarVar,
}


def make_var_from_value(value: Any,
                        need_guard_check: bool,
                        extract_code_at_start: str = "") -> Variable:
    return ty2var[type(value)].from_value(value, need_guard_check,
                                          extract_code_at_start)


__all__ = ['make_var_from_value', 'Variable']
