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

NodeArgs = Union[BaseArgumentTypes, torch.fx.Node]


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
    mark_written_fn: Callable[[], None]
    fake_mode: torch._subclasses.FakeTensorMode
    example_inputs: list[tuple[str, torch.Tensor]]

    def __init__(self, root: torch.nn.Module,
                 mark_written_fn: Callable[[], None]) -> None:
        self.root = root
        self.result_graph = torch.fx.Graph(root)
        self.mark_written_fn = mark_written_fn
        self.fake_mode = torch._subclasses.FakeTensorMode()
        self.example_inputs = []

    def create_node(
        self,
        kind: str,
        target: torch.fx.node.Target,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
        name: Optional[str] = None,
        type_expr: Optional[Any] = None,
    ) -> torch.fx.Node:
        self.mark_written_fn()
        return self.result_graph.create_node(kind, target, args, kwargs, name,
                                             type_expr)

    def create_input(
        self,
        value: torch.Tensor,
        target: torch.fx.node.Target,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
        name: Optional[str] = None,
        type_expr: Optional[Any] = None,
    ) -> torch.fx.Node:
        fake_tensor = self.fake_mode.from_tensor(value, static_shapes=True)
        self.mark_written_fn()
        self.example_inputs.append((fake_tensor, name))
        return self.create_node("placeholder", target, args, kwargs, name,
                                type_expr)

    def set_output_nodes(self, output_nodes: list[torch.fx.Node]) -> None:
        for node in self.result_graph.nodes:
            assert node.op != "output"
        self.result_graph.output(tuple(output_nodes))

    def compile(
        self,
    ) -> Any:  # heheda: shoud be Callable[..., Any], but I cannot pass mypy check
        model = torch.fx.GraphModule(self.root, self.result_graph)
        model.recompile()
        with NO_LD_PRELOAD_CTX():
            compiled_fn = backend_compile(model,
                                          [x[0] for x in self.example_inputs])
        assert callable(compiled_fn)
        # TODO: add backend compiler
        return compiled_fn

    def get_inputs(self) -> list[torch.fx.Node]:
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