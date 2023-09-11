from typing import Any, Union
import torch
from .base import Variable
from .scalar import ScalarVar
from typing import Optional
from .tensor import TensorVar
from .torch_module import TorchModuleVar
from .null import NullVar
from .none import NoneVar
from ..fx_graph import FxGraph
from ..utils import NullObject

ty2var: dict[type[Any], type[Variable]] = {
    float: ScalarVar,
    int: ScalarVar,
    torch.Tensor: TensorVar,
    NullObject: NullVar,
    type(None): NoneVar
}

CONST_TYPES = Union[int, float, bool, str, NullObject, None]


def make_var_from_value(value: Any,
                        need_guard_check: bool,
                        fx_graph: Optional[FxGraph] = None,
                        extract_code_at_start: str = "") -> Variable:
    if isinstance(value, torch.nn.Module):
        return TorchModuleVar.from_value(value, need_guard_check, fx_graph,
                                         extract_code_at_start)
    else:
        return ty2var[type(value)].from_value(value, need_guard_check, fx_graph,
                                              extract_code_at_start)


__all__ = [
    'make_var_from_value', 'Variable', 'ScalarVar', 'TensorVar',
    'TorchModuleVar', 'NullVar', 'NoneVar'
]
