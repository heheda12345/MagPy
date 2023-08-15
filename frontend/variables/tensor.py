# def proxy_args_kwargs(args, kwargs):
#     try:
#         proxy_args = tuple(arg.as_proxy() for arg in args)
#         proxy_kwargs = {key: arg.as_proxy() for key, arg in kwargs.items()}
#         return proxy_args, proxy_kwargs
#     except NotImplementedError as e:
#         from .exc import unimplemented
#         from .variables.base import typestr

#         raise unimplemented(
#             f"call_function args: {typestr(*args)} {typestr(*list(kwargs.values()))}"
#         ) from e

import torch
import torch.fx
from typing import Union

_counter = 0


def new_name() -> str:
    global _counter
    _counter += 1
    return f"__tensor_var_{_counter}"


class TensorVar:
    proxy: torch.fx.Proxy

    def __init__(self, px: torch.fx.Proxy):
        if isinstance(px, torch.fx.Proxy):
            self.proxy = px
        else:
            self.proxy = tracer.create_proxy("placeholder", new_name(), (), {})

    @classmethod
    def from_tensor(cls,
                    x: torch.Tensor,
                    tracer: torch.fx.Tracer,
                    name: str = ""):
        name = name or new_name()
        proxy = tracer.create_proxy("placeholder", name, (), {})
        return cls(proxy)


if __name__ == '__main__':
    a = torch.full((1,), 1.0)
    b = torch.full((1,), 2.0)
    c = torch.full((1,), 3.0)

    result_graph = torch.fx.Graph()
    tracer = torch.fx.proxy.GraphAppendingTracer(result_graph)

    var_a = TensorVar.from_tensor(a, tracer)
    var_b = TensorVar.from_tensor(b, tracer)
    var_c = TensorVar.from_tensor(c, tracer)

    var_d = TensorVar(var_a.proxy + var_b.proxy)
    var_e = TensorVar(var_d.proxy + var_c.proxy)
    result_graph.output(var_e.proxy.node)
    print(result_graph)
    model = torch.fx.GraphModule(torch.nn.Module(), result_graph)
    model.recompile()
    print(model)
    print(
        model(torch.full((1,), 4.0), torch.full((1,), 5.0), torch.full((1,),
                                                                       6.0)))
