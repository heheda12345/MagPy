from typing import Any, Union, Optional, Tuple, TYPE_CHECKING, Callable
from types import ModuleType
import torch
from .base import Variable
from .scalar import ScalarVar
from .tensor import TensorVar, TorchParamVar, TorchSizeVar, TorchDtypeVar
from .torch_module import TorchModuleVar, TorchSequentialVar, TorchModuleListVar
from .any_ import AnyVar
from .const import NullVar, NoneVar, SliceVar, ModuleVar, FunctionVar, RangeVar
from .tuple_ import TupleVar
from .set_ import SetVar
from .list_ import ListVar
from .dict_ import DictVar
from ..fx_graph import FxGraph
from ..utils import NullObject, UnknownTypeError
from ..store_pos import StorePos

ty2var: dict[type[Any], type[Variable]] = {
    float: ScalarVar,
    int: ScalarVar,
    str: ScalarVar,
    torch.Tensor: TensorVar,
    NullObject: NullVar,
    type(None): NoneVar,
    slice: SliceVar,
    torch.nn.Parameter: TorchParamVar,
    tuple: TupleVar,
    list: ListVar,
    set: SetVar,
    torch.Size: TorchSizeVar,
    torch.dtype: TorchDtypeVar,
    dict: DictVar,
}

CONST_TYPES = Union[int, float, bool, str, NullObject, None, slice]


def make_var_from_value(value: Any,
                        need_guard_check: bool,
                        get_or_make_var: Callable[
                            [Any, bool, Optional[FxGraph], list[StorePos]],
                            Variable],
                        fx_graph: Optional[FxGraph] = None,
                        extract_code_at_start: list[StorePos] = []) -> Variable:
    if type(value) in ty2var:
        return ty2var[type(value)].from_value(value, need_guard_check,
                                              get_or_make_var, fx_graph,
                                              extract_code_at_start)
    elif isinstance(value, torch.nn.Module):
        return TorchModuleVar.from_value(value, need_guard_check,
                                         get_or_make_var, fx_graph,
                                         extract_code_at_start)
    elif isinstance(value, ModuleType):
        return ModuleVar.from_value(value, need_guard_check, get_or_make_var,
                                    fx_graph, extract_code_at_start)
    elif callable(value):
        return FunctionVar.from_value(value, need_guard_check, get_or_make_var,
                                      fx_graph, extract_code_at_start)
    elif isinstance(value, range):
        return RangeVar.from_value(value, need_guard_check, get_or_make_var,
                                   fx_graph, extract_code_at_start)
    else:
        return AnyVar.from_value(value, need_guard_check, get_or_make_var,
                                 fx_graph, extract_code_at_start)


__all__ = [
    'make_var_from_value', 'Variable', 'ScalarVar', 'TensorVar',
    'TorchModuleVar', 'NullVar', 'NoneVar', "ModuleVar", "FunctionVar",
    "TorchParamVar", "AnyVar"
]
