from typing import Any, Union, Optional, Tuple, TYPE_CHECKING, Callable
from types import ModuleType, CodeType, CellType, MappingProxyType
import torch
import numpy as np
from collections import OrderedDict
from .base import Variable, HelperFunctions
from .scalar import ScalarVar, NumpyScalarVar
from .tensor import TensorVar, TorchParamVar, TorchSizeVar, TorchDtypeVar, TorchDeviceVar, TorchLayoutVar
from .torch_module import TorchModuleVar, TorchSequentialVar, TorchModuleListVar
from .any_ import AnyVar
from .const import NullVar, NoneVar, SliceVar, ModuleVar, FunctionVar, RangeVar, CodeVar, EllipsisVar
from .iterator import IteratorVar, RangeIterVar
from .tuple_ import TupleVar
from .set_ import SetVar
from .list_ import ListVar, NdarrayVar
from .dict_ import DictVar, OrderedDictVar
from .builtin_types import CellVar, MappingProxyVar
from ..fx_graph import FxGraph
from ..utils import NullObject, UnknownTypeError, is_structseq
from ..store_pos import StorePos

ty2var: dict[type[Any], type[Variable]] = {
    float: ScalarVar,
    int: ScalarVar,
    str: ScalarVar,
    bool: ScalarVar,
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
    torch.device: TorchDeviceVar,
    torch.layout: TorchLayoutVar,
    dict: DictVar,
    CodeType: CodeVar,
    OrderedDict: OrderedDictVar,
    np.ndarray: NdarrayVar,
}

CONST_TYPES = Union[int, float, bool, str, NullObject, None, slice]


def make_var_from_value(
        value: Any,
        need_guard_check: bool,
        helper_functions: HelperFunctions,
        fx_graph: Optional[FxGraph] = None,
        extract_code_at_start: Optional[list[StorePos]] = None) -> Variable:
    if extract_code_at_start is None:
        extract_code_at_start = []
    if type(value) in ty2var:
        return ty2var[type(value)].from_value(value, need_guard_check,
                                              helper_functions, fx_graph,
                                              extract_code_at_start)
    elif isinstance(value, torch.nn.Module):
        return TorchModuleVar.from_value(value, need_guard_check,
                                         helper_functions, fx_graph,
                                         extract_code_at_start)
    elif isinstance(value, ModuleType):
        return ModuleVar.from_value(value, need_guard_check, helper_functions,
                                    fx_graph, extract_code_at_start)
    elif callable(value):
        return FunctionVar.from_value(value, need_guard_check, helper_functions,
                                      fx_graph, extract_code_at_start)
    elif isinstance(value, range):
        return RangeVar.from_value(value, need_guard_check, helper_functions,
                                   fx_graph, extract_code_at_start)
    elif isinstance(value, type(range(0).__iter__())):
        return RangeIterVar.from_value(value, need_guard_check,
                                       helper_functions, fx_graph,
                                       extract_code_at_start)
    elif isinstance(value, CellType):
        return CellVar.from_value(value, need_guard_check, helper_functions,
                                  fx_graph, extract_code_at_start)
    elif isinstance(value, np.generic):
        return NumpyScalarVar.from_value(value, need_guard_check,
                                         helper_functions, fx_graph,
                                         extract_code_at_start)
    elif is_structseq(value):
        return TupleVar.from_value(value, need_guard_check, helper_functions,
                                   fx_graph, extract_code_at_start)
    elif type(value) == MappingProxyType:
        return MappingProxyVar.from_value(value, need_guard_check,
                                          helper_functions, fx_graph,
                                          extract_code_at_start)
    elif isinstance(value, type(Ellipsis)):
        return EllipsisVar.from_value(value, need_guard_check, helper_functions,
                                      fx_graph, extract_code_at_start)
    else:
        # NOTE: use any instead of iteartor_var to represent iterator with unknown source due to the hardness of getting iterable and num_iters
        print("generate any for", value, type(value), extract_code_at_start)
        return AnyVar.from_value(value, need_guard_check, helper_functions,
                                 fx_graph, extract_code_at_start)


__all__ = [
    'make_var_from_value', 'Variable', 'ScalarVar', 'TensorVar',
    'TorchModuleVar', 'NullVar', 'NoneVar', "ModuleVar", "FunctionVar",
    "TorchParamVar", "AnyVar", "IteratorVar", "RangeIterVar", "HelperFunctions"
]
