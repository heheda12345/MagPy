from typing import Any, Callable, Dict, Optional, Tuple, Union
import operator
import torch
import torch.fx


def fx_graph_functions() -> set[Callable[..., Any]]:
    fns: set[Callable[..., Any]] = {
        operator.pos,
        operator.neg,
        operator.not_,
        operator.invert,
        operator.pow,
        operator.mul,
        operator.matmul,
        operator.floordiv,
        operator.truediv,
        operator.mod,
        operator.add,
        operator.sub,
        operator.getitem,
        operator.lshift,
        operator.rshift,
        operator.and_,
        operator.or_,
        operator.xor,
        # operator.ipow,
        # operator.imul,
        # operator.imatmul,
        # operator.ifloordiv,
        # operator.itruediv,
        # operator.imod,
        # operator.iadd,
        # operator.isub,
        # operator.ilshift,
        # operator.irshift,
        # operator.iand,
        # operator.ixor,
        # operator.ior,
    }
    return fns


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


class FxGraph:
    root: torch.nn.Module
    result_graph: torch.fx.Graph
    tracer: torch.fx.Tracer

    def __init__(self, root: torch.nn.Module) -> None:
        self.root = root
        self.result_graph = torch.fx.Graph(root)
        self.tracer = torch.fx.proxy.GraphAppendingTracer(self.result_graph)

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
        return self.tracer.create_proxy(kind, target, args, kwargs, name,
                                        type_expr, proxy_factory_fn)

    def compile(
        self, outputs: list[torch.fx.Proxy]
    ) -> Any:  # heheda: shoud be Callable[..., Any], but I cannot pass mypy check
        output_nodes = tuple((x.node for x in outputs))
        self.result_graph.output(output_nodes)
        print("fx graph:", self.result_graph)
        model = torch.fx.GraphModule(self.root, self.result_graph)
        model.recompile()
        assert callable(model)
        # TODO: add backend compiler
        return model

    def get_inputs(self) -> list[torch.fx.Proxy]:
        return [x for x in self.result_graph.nodes if x.op == "placeholder"]


frame_root: dict[int, torch.nn.Module] = {}


def set_frame_root(frame_id: int, root: Any) -> None:
    if frame_id in frame_root:
        return
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