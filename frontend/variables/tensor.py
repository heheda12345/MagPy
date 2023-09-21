from typing import Optional, Any
import torch
import torch.fx

from frontend.pycode_generator import GuardFnCodegen, GraphFnCodegen
from .base import Variable
from ..pycode_writer import new_name
from ..fx_graph import FxGraph, ProxyArgs
from ..store_pos import StorePos


class TensorVar(Variable):
    proxy: torch.fx.Proxy
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

    def __init__(self,
                 proxy: torch.fx.Proxy,
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
                 extract_code_at_start: list[StorePos] = []) -> None:
        super().__init__(need_guard_check, extract_code_at_start)
        self.proxy = proxy
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

    @classmethod
    def from_tensor_and_proxy(
            cls,
            tensor: torch.Tensor,
            proxy: torch.fx.Proxy,
            need_guard_check: bool,
            extract_code_at_start: list[StorePos] = []) -> 'TensorVar':
        var = cls(proxy, tensor.dtype, tensor.device, tensor.layout,
                  tensor.ndim, tensor.requires_grad, tensor.is_quantized,
                  tensor.is_sparse, type(tensor),
                  need_guard_check, tensor.size(), tensor.stride(),
                  tensor.is_contiguous(), extract_code_at_start)
        proxy.node.meta["var"] = var
        return var

    @classmethod
    def from_value(cls,
                   value: torch.Tensor,
                   need_guard_check: bool,
                   fx_graph: Optional[FxGraph] = None,
                   extract_code_at_start: list[StorePos] = []) -> 'TensorVar':
        assert fx_graph is not None
        name = new_name('tensor')
        assert len(extract_code_at_start) > 0
        proxy = fx_graph.create_proxy("placeholder", name, (), {}, name)
        var = cls.from_tensor_and_proxy(value, proxy, need_guard_check,
                                        extract_code_at_start)
        return var

    def as_proxy(self) -> ProxyArgs:
        return self.proxy

    def tensor_guard_check(self, value: torch.Tensor) -> bool:
        return isinstance(value, torch.Tensor) and self.dtype == value.dtype and self.device == value.device and \
            self.layout == value.layout and self.ndim == value.ndim and \
            self.requires_grad == value.requires_grad and \
            self.is_quantized == value.is_quantized and \
            self.is_sparse == value.is_sparse and \
            self.class_type == type(value) and \
            hasattr(value, 'size') and self.size == value.size() and \
            hasattr(value, 'stride') and self.stride == value.stride() and \
            hasattr(value, 'is_contiguous') and self.is_contiguous == value.is_contiguous()

    def make_guard_inner(self, codegen: "GuardFnCodegen",
                         pos: StorePos) -> None:
        name_in_codegen = codegen.add_var(self)
        codegen.add_check(f"{name_in_codegen}.tensor_guard_check({pos})")

    def make_output(self, name_in_graph_fn: str, store_pos: StorePos,
                    codegen: "GraphFnCodegen") -> None:
        name_in_graph_output = codegen.add_graph_output(self.proxy)
        codegen.output(name_in_graph_fn, store_pos, name_in_graph_output)


class TorchParamVar(Variable):
    param: torch.nn.Parameter

    def __init__(self,
                 param: torch.nn.Parameter,
                 need_guard_check: bool,
                 extract_code_at_start: list[StorePos] = []) -> None:
        super().__init__(need_guard_check, extract_code_at_start)
        assert len(extract_code_at_start) > 0
        self.param = param

    @classmethod
    def from_value(
            cls,
            value: torch.nn.Parameter,
            need_guard_check: bool,
            _fx_graph: Optional[FxGraph] = None,
            extract_code_at_start: list[StorePos] = []) -> "TorchParamVar":
        return cls(value, need_guard_check, extract_code_at_start)

    def make_guard_inner(self, codegen: "GuardFnCodegen",
                         pos: StorePos) -> None:
        codegen.add_check(f"id({pos}) == {id(self.param)}")

    def make_output(self, name_in_graph_fn: str, store_pos: StorePos,
                    codegen: "GraphFnCodegen") -> None:
        codegen.output(name_in_graph_fn, store_pos,
                       str(self.extract_code_at_start))

    def as_proxy(self) -> "ProxyArgs":
        raise ValueError("TorchParamVar.as_proxy should not be called")
