from typing import Optional, Any, Callable
import torch
import torch.fx

from frontend.pycode_generator import GuardFnCodegen, GraphFnCodegen
from .base import Variable
from .tuple_ import TupleVar
from ..pycode_writer import new_name
from ..fx_graph import FxGraph, NodeArgs
from ..store_pos import StorePos


class TensorVar(Variable):
    fx_node: torch.fx.Node
    dtype: torch.dtype
    device: torch.device
    layout: torch.layout
    ndim: int
    requires_grad: bool
    is_quantized: bool
    is_sparse: bool
    class_type: type
    size: Optional[tuple[Optional[int], ...]]
    stride: Optional[tuple[Optional[int], ...]]
    is_contiguous: Optional[bool]
    idx: int

    def __init__(self,
                 tensor: torch.Tensor,
                 fx_node: torch.fx.Node,
                 dtype: torch.dtype,
                 device: torch.device,
                 layout: torch.layout,
                 ndim: int,
                 requires_grad: bool,
                 is_quantized: bool,
                 is_sparse: bool,
                 class_type: type,
                 need_guard_check: bool,
                 size: Optional[tuple[Optional[int], ...]] = None,
                 stride: Optional[tuple[Optional[int], ...]] = None,
                 is_contiguous: Optional[bool] = None,
                 idx: int = 0,
                 extract_code_at_start: list[StorePos] = []) -> None:
        super().__init__(need_guard_check, tensor, extract_code_at_start)
        self.fx_node = fx_node
        self.dtype = dtype
        self.device = device
        self.layout = layout
        self.ndim = ndim
        self.size = size
        self.stride = stride
        self.requires_grad = requires_grad
        self.is_quantized = is_quantized
        self.is_contiguous = is_contiguous
        self.is_sparse = is_sparse
        self.class_type = class_type
        self.idx = idx

    @classmethod
    def from_tensor_and_node(
            cls,
            tensor: torch.Tensor,
            fx_node: torch.fx.Node,
            need_guard_check: bool,
            extract_code_at_start: list[StorePos] = []) -> 'TensorVar':
        var = cls(tensor, fx_node, tensor.dtype, tensor.device, tensor.layout,
                  tensor.ndim, tensor.requires_grad, tensor.is_quantized,
                  tensor.is_sparse, type(tensor), need_guard_check,
                  tensor.size(), tensor.stride(), tensor.is_contiguous(),
                  id(tensor), extract_code_at_start)
        fx_node.meta["var"] = var
        return var

    @classmethod
    def from_value(cls,
                   value: torch.Tensor,
                   need_guard_check: bool,
                   _get_or_make_var: Callable[
                       [Any, bool, Optional[FxGraph], list[StorePos]],
                       Variable],
                   fx_graph: Optional[FxGraph] = None,
                   extract_code_at_start: list[StorePos] = []) -> 'TensorVar':
        assert fx_graph is not None
        name = new_name('tensor')
        assert len(extract_code_at_start) > 0
        fx_node = fx_graph.create_input(value, name, (), {}, name)
        var = cls.from_tensor_and_node(value, fx_node, need_guard_check,
                                       extract_code_at_start)
        return var

    def as_fx_node(self) -> NodeArgs:
        return self.fx_node

    def tensor_guard_check(self, value: torch.Tensor) -> bool:
        return isinstance(value, torch.Tensor) and self.dtype == value.dtype and self.device == value.device and \
            self.layout == value.layout and self.ndim == value.ndim and \
            self.requires_grad == value.requires_grad and \
            self.is_quantized == value.is_quantized and \
            self.is_sparse == value.is_sparse and \
            self.class_type == type(value) and \
            hasattr(value, 'size') and self.size == value.size() # and \
        # hasattr(value, 'stride') and self.stride == value.stride() and \
        # hasattr(value, 'is_contiguous') and self.is_contiguous == value.is_contiguous()

    def make_guard_inner(self, codegen: "GuardFnCodegen",
                         pos: StorePos) -> None:
        name_in_codegen = codegen.add_var(self)
        codegen.add_check(f"{name_in_codegen}.tensor_guard_check({pos})")

    def make_output_inner(self, name_in_graph_fn: str, store_pos: StorePos,
                          codegen: "GraphFnCodegen", in_return: bool,
                          idx: int) -> None:
        name_in_graph_output = codegen.add_graph_output(self.fx_node)
        codegen.output(name_in_graph_fn, store_pos, name_in_graph_output,
                       in_return, idx)


class TorchParamVar(Variable):

    def __init__(self,
                 param: torch.nn.Parameter,
                 need_guard_check: bool,
                 extract_code_at_start: list[StorePos] = []) -> None:
        super().__init__(need_guard_check, param, extract_code_at_start)
        assert len(extract_code_at_start) > 0

    @classmethod
    def from_value(
            cls,
            value: torch.nn.Parameter,
            need_guard_check: bool,
            _get_or_make_var: Callable[
                [Any, bool, Optional[FxGraph], list[StorePos]], Variable],
            _fx_graph: Optional[FxGraph] = None,
            extract_code_at_start: list[StorePos] = []) -> "TorchParamVar":
        return cls(value, need_guard_check, extract_code_at_start)

    def make_guard_inner(self, codegen: "GuardFnCodegen",
                         pos: StorePos) -> None:
        codegen.add_id_check(f"id({pos}) == {id(self.obj)}", self.obj)

    def make_output_inner(self, name_in_graph_fn: str, store_pos: StorePos,
                          codegen: "GraphFnCodegen", in_return: bool,
                          idx: int) -> None:
        codegen.output(name_in_graph_fn, store_pos,
                       str(self.extract_code_at_start), in_return, idx)

    def as_fx_node(self) -> "NodeArgs":
        raise ValueError("TorchParamVar.as_fx_node should not be called")


class TorchSizeVar(TupleVar):

    def make_output_inner(self, name_in_graph_fn: str, store_pos: StorePos,
                          codegen: "GraphFnCodegen", in_return: bool,
                          idx: int) -> None:
        tuple_name = name_in_graph_fn + "_tuple"
        super().make_output_inner(tuple_name, store_pos, codegen, False, idx)
        codegen.output(name_in_graph_fn, store_pos, f"torch.Size({tuple_name})",
                       in_return, idx)
