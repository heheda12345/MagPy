from typing import Any, Callable, Dict, Optional, Tuple, Union
import torch
import torch.fx
import torch._inductor.compile_fx
from .utils import NO_LD_PRELOAD_CTX
from . import config

BaseArgumentTypes = Union[
    str,
    int,
    float,
    bool,
    complex,
    #   torch.dtype, torch.Tensor, torch.device, torch.memory_format,
    #   torch.layout, torch._ops.OpOverload
]

ProxyArgs = Union[BaseArgumentTypes, torch.fx.Proxy]


def backend_compile(gm: torch.fx.GraphModule,
                    example_inputs: list[torch.Tensor]) -> Any:
    if callable(config.backend):
        return config.backend(gm, example_inputs)
    elif config.backend == 'eager':
        return gm
    elif config.backend == 'inductor':
        return torch._inductor.compile_fx.compile_fx(gm, example_inputs)
    else:
        raise RuntimeError(f"Unknown backend: {config.backend}")


class FxGraph:
    root: torch.nn.Module
    result_graph: torch.fx.Graph
    tracer: torch.fx.Tracer
    mark_written_fn: Callable[[], None]
    fake_mode: torch._subclasses.FakeTensorMode
    example_inputs: list[tuple[str, torch.Tensor]]

    def __init__(self, root: torch.nn.Module,
                 mark_written_fn: Callable[[], None]) -> None:
        self.root = root
        self.result_graph = torch.fx.Graph(root)
        self.tracer = torch.fx.proxy.GraphAppendingTracer(self.result_graph)
        self.mark_written_fn = mark_written_fn
        self.fake_mode = torch._subclasses.FakeTensorMode()
        self.example_inputs = []

    def create_proxy(
        self,
        kind: str,
        target: torch.fx.node.Target,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
        name: Optional[str] = None,
        type_expr: Optional[Any] = None,
        proxy_factory_fn: Optional[Callable[[torch.fx.Node],
                                            torch.fx.Proxy]] = None
    ) -> torch.fx.Proxy:
        self.mark_written_fn()
        return self.tracer.create_proxy(kind, target, args, kwargs, name,
                                        type_expr, proxy_factory_fn)

    def create_input(
        self,
        value: torch.Tensor,
        target: torch.fx.node.Target,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
        name: Optional[str] = None,
        type_expr: Optional[Any] = None,
        proxy_factory_fn: Optional[Callable[[torch.fx.Node],
                                            torch.fx.Proxy]] = None
    ) -> torch.fx.Proxy:
        fake_tensor = self.fake_mode.from_tensor(value, static_shapes=True)
        self.mark_written_fn()
        self.example_inputs.append((fake_tensor, name))
        return self.create_proxy("placeholder", target, args, kwargs, name,
                                 type_expr, proxy_factory_fn)

    def compile(
        self, outputs: list[torch.fx.Proxy]
    ) -> Any:  # heheda: shoud be Callable[..., Any], but I cannot pass mypy check
        output_nodes = tuple((x.node for x in outputs))
        self.result_graph.output(output_nodes)
        print("fx graph:", self.result_graph)
        model = torch.fx.GraphModule(self.root, self.result_graph)
        model.recompile()
        with NO_LD_PRELOAD_CTX():
            compiled_fn = backend_compile(model,
                                          [x[0] for x in self.example_inputs])
        assert callable(compiled_fn)
        # TODO: add backend compiler
        return compiled_fn

    def get_inputs(self) -> list[torch.fx.Proxy]:
        return [x for x in self.result_graph.nodes if x.op == "placeholder"]


frame_root: dict[int, torch.nn.Module] = {}


def set_frame_root(frame_id: int, root: Any) -> None:
    if isinstance(root, torch.nn.Module):
        root_module = root
    elif hasattr(root, '__self__') and isinstance(root.__self__,
                                                  torch.nn.Module):
        root_module = root.__self__
    else:
        root_module = torch.nn.Module()

    frame_root[frame_id] = root_module


def get_frame_root(frame_id: int) -> Any:
    return frame_root[frame_id]


def is_leaf_module(m: torch.nn.Module) -> bool:
    return ((m.__module__.startswith("torch.nn") or
             m.__module__.startswith("torch.ao.nn")) and
            not isinstance(m, torch.nn.Sequential))


def reset() -> None:
    global frame_root
    frame_root = {}