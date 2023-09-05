from typing import Optional, Any
import torch
import torch.fx

from frontend.pycode_generator import GuardFnCodegen, GraphFnCodegen
from .base import Variable
from ..pycode_writer import new_name
from ..fx_graph import FxGraph


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
                 extract_code_at_start: str = "") -> None:
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
    def from_tensor_and_proxy(cls,
                              tensor: torch.Tensor,
                              proxy: torch.fx.Proxy,
                              need_guard_check: bool,
                              extract_code_at_start: str = "") -> 'TensorVar':
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
                   fx_graph: 'FxGraph',
                   extract_code_at_start: str = "") -> 'TensorVar':
        name = new_name('tensor')
        assert extract_code_at_start != ""
        proxy = fx_graph.create_proxy("placeholder", name, (), {}, name)
        var = cls.from_tensor_and_proxy(value, proxy, need_guard_check,
                                        extract_code_at_start)
        return var

    def as_proxy(self) -> torch.fx.Proxy:
        return self.proxy

    def guard_check(self, value: torch.Tensor) -> bool:
        print("checking", value)
        return isinstance(value, torch.Tensor) and self.dtype == value.dtype and self.device == value.device and \
            self.layout == value.layout and self.ndim == value.ndim and \
            self.requires_grad == value.requires_grad and \
            self.is_quantized == value.is_quantized and \
            self.is_sparse == value.is_sparse and \
            self.class_type == type(value) and \
            hasattr(value, 'size') and self.size == value.size() and \
            hasattr(value, 'stride') and self.stride == value.stride() and \
            hasattr(value, 'is_contiguous') and self.is_contiguous == value.is_contiguous()

    def make_guard_inner(self, codegen: GuardFnCodegen) -> None:
        name_in_codegen = codegen.add_var(self)
        codegen.add_check(
            f"{name_in_codegen}.guard_check({self.extract_code_at_start})")

    def make_output(self, target_name: str, codegen: GraphFnCodegen) -> None:
        name_in_graph_output = codegen.add_graph_output(self.proxy)
        codegen.output(target_name, name_in_graph_output)