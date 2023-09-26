from typing import Any, Union, Optional, Tuple, TYPE_CHECKING
from types import ModuleType
import torch
from .base import Variable
from .scalar import ScalarVar
from .tensor import TensorVar, TorchParamVar
from .torch_module import TorchModuleVar
from .const import NullVar, NoneVar, SliceVar, ModuleVar, FunctionVar, ObjectSrc
from .tuple_ import TupleVar
from .list_ import ListVar
from .dict_ import DictVar
from ..fx_graph import FxGraph
from ..utils import NullObject, UnknownTypeError
from ..store_pos import StorePos
if TYPE_CHECKING:
    from ..object_table import ReadOnlyObjectTable

ty2var: dict[type[Any], type[Variable]] = {
    float: ScalarVar,
    int: ScalarVar,
    torch.Tensor: TensorVar,
    NullObject: NullVar,
    type(None): NoneVar,
    slice: SliceVar,
    torch.nn.Parameter: TorchParamVar,
    tuple: TupleVar,
    list: ListVar,
    dict: DictVar,
}

CONST_TYPES = Union[int, float, bool, str, NullObject, None, slice]


def make_var_from_value(value: Any,
                        need_guard_check: bool,
                        object_table: 'ReadOnlyObjectTable',
                        fx_graph: Optional[FxGraph] = None,
                        extract_code_at_start: list[StorePos] = []) -> Variable:
    if type(value) in ty2var:
        return ty2var[type(value)].from_value(value, need_guard_check,
                                              object_table, fx_graph,
                                              extract_code_at_start)
    elif isinstance(value, torch.nn.Module):
        return TorchModuleVar.from_value(value, need_guard_check, object_table,
                                         fx_graph, extract_code_at_start)
    elif isinstance(value, ModuleType):
        return ModuleVar.from_value(value, need_guard_check, object_table,
                                    fx_graph, extract_code_at_start)
    elif callable(value):
        return FunctionVar.from_value(value, need_guard_check, object_table,
                                      fx_graph, extract_code_at_start)
    raise UnknownTypeError(type(value))


__all__ = [
    'make_var_from_value', 'Variable', 'ScalarVar', 'TensorVar',
    'TorchModuleVar', 'NullVar', 'NoneVar', "ModuleVar", "FunctionVar",
    "TorchParamVar", "ObjectSrc"
]
