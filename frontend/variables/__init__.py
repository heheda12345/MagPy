from typing import Any, Union, Optional
from types import ModuleType
import torch
from .base import Variable
from .scalar import ScalarVar
from .tensor import TensorVar
from .torch_module import TorchModuleVar
from .const import NullVar, NoneVar, SliceVar, ModuleVar
from ..fx_graph import FxGraph
from ..utils import NullObject, UnknownTypeError

ty2var: dict[type[Any], type[Variable]] = {
    float: ScalarVar,
    int: ScalarVar,
    torch.Tensor: TensorVar,
    NullObject: NullVar,
    type(None): NoneVar,
    slice: SliceVar,
    ModuleType: ModuleVar
}

CONST_TYPES = Union[int, float, bool, str, NullObject, None, slice]


def make_var_from_value(value: Any,
                        need_guard_check: bool,
                        fx_graph: Optional[FxGraph] = None,
                        extract_code_at_start: str = "") -> Variable:
    if isinstance(value, torch.nn.Module):
        return TorchModuleVar.from_value(value, need_guard_check, fx_graph,
                                         extract_code_at_start)
    else:
        if type(value) not in ty2var:
            raise UnknownTypeError(type(value))
        return ty2var[type(value)].from_value(value, need_guard_check, fx_graph,
                                              extract_code_at_start)


__all__ = [
    'make_var_from_value', 'Variable', 'ScalarVar', 'TensorVar',
    'TorchModuleVar', 'NullVar', 'NoneVar'
]
