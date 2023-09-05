from typing import Any, cast
import torch
from .base import Variable
from .scalar import ScalarVar
from .tensor import TensorVar
from ..fx_graph import FxGraph

ty2var: dict[type[Any], type[Variable]] = {
    float: ScalarVar,
    int: ScalarVar,
    torch.Tensor: TensorVar,
}


def make_var_from_value(value: Any,
                        need_guard_check: bool,
                        fx_graph: FxGraph,
                        extract_code_at_start: str = "") -> Variable:
    return ty2var[type(value)].from_value(value, need_guard_check, fx_graph,
                                          extract_code_at_start)


__all__ = ['make_var_from_value', 'Variable', 'ScalarVar', 'TensorVar']
