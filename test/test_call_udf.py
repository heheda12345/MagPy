from frontend.compile import compile, reset
from frontend.utils import add_force_graph_break
from frontend.c_api import get_next_frame_id
from common.checker import run_and_check, HIT, MISS
import torch


class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x) - 4.0


def func(a):
    return a + 1


def call_func(a):
    b = func(a) + 1
    return b


def call_model(model, a):
    b = model(a) + 1
    return b


class Model2(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.layer = Model()

    def forward(self, x):
        return self.layer(x) * 2.0


class ParamModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.full((1, 1), 5.0))

    def forward(self, x):
        return x * self.param


def func_kw(a, b, c=1.0):
    return a + b + c


def call_function_kw(a, b, c):
    x = func_kw(a, b, c=2.0)
    y = func_kw(a, b)
    z = func_kw(a, b, c)
    return x + y + z


def test_call_ud_func_break(caplog):
    reset()
    compiled_call_func = compile(call_func)
    a = torch.full((1, 1), 1.0)
    result = call_func(a)
    add_force_graph_break(get_next_frame_id() + 1, 1)
    run_and_check(compiled_call_func, [MISS, MISS], 4, caplog, result, a)
    run_and_check(compiled_call_func, [HIT, HIT, HIT, HIT], 4, caplog, result,
                  a)


def test_call_ud_model_break(caplog):
    reset()
    model = Model()
    a = torch.full((1, 1), 1.0)
    result = call_model(model, a)
    compiled_call_model = compile(call_model)
    add_force_graph_break(get_next_frame_id() + 1, 4)
    run_and_check(compiled_call_model, [MISS, MISS], 4, caplog, result, model,
                  a)
    run_and_check(compiled_call_model, [HIT, HIT, HIT, HIT], 4, caplog, result,
                  model, a)

    model2 = Model2()
    result = model2(a)
    compiled_model = compile(model2)
    run_and_check(compiled_model, [MISS, MISS, HIT], 7, caplog, result, a)
    run_and_check(compiled_model, [HIT, HIT, HIT, HIT], 7, caplog, result, a)


def test_call_param_module_break(caplog):
    reset()
    model = ParamModel()
    a = torch.full((1, 1), 2.0)
    result = call_model(model, a)
    compiled_model = compile(call_model)
    add_force_graph_break(
        get_next_frame_id(),
        2)  # not need +1 because call_model func is called before
    run_and_check(compiled_model, [MISS, MISS], 4, caplog, result, model, a)
    run_and_check(compiled_model, [HIT, HIT, HIT, HIT], 4, caplog, result,
                  model, a)


def test_call_ud_func_scalar(caplog):
    reset()
    compiled_call_func = compile(call_func)
    a = 1.0
    result = call_func(a)
    run_and_check(compiled_call_func, [MISS, MISS], 1, caplog, result, a)
    run_and_check(compiled_call_func, [HIT], 1, caplog, result, a)


def test_call_ud_func_tensor(caplog):
    reset()
    compiled_call_func = compile(call_func)
    a = torch.tensor(1.0)
    result = call_func(a)
    run_and_check(compiled_call_func, [MISS, MISS], 1, caplog, result, a)
    run_and_check(compiled_call_func, [HIT], 1, caplog, result, a)


def test_call_ud_module_external(caplog):
    reset()
    model = Model()
    a = torch.full((1, 1), 1.0)
    result = call_model(model, a)
    compiled_call_model = compile(call_model)
    run_and_check(compiled_call_model, [MISS, MISS], 1, caplog, result, model,
                  a)
    run_and_check(compiled_call_model, [HIT], 1, caplog, result, model, a)

    model2 = Model2()
    result = model2(a)
    compiled_model = compile(model2)
    run_and_check(compiled_model, [MISS, MISS], 2, caplog, result, a)
    run_and_check(compiled_model, [HIT], 2, caplog, result, a)


def test_call_param_module(caplog):
    reset()
    model = ParamModel()
    a = torch.full((1, 1), 2.0)
    result = call_model(model, a)
    compiled_model = compile(call_model)
    run_and_check(compiled_model, [MISS, MISS], 1, caplog, result, model, a)
    run_and_check(compiled_model, [HIT], 1, caplog, result, model, a)


def test_call_function_kw(caplog):
    reset()
    a = 1.0
    b = 2.0
    c = 3.0
    result = call_function_kw(a, b, c)
    compiled_call_func = compile(call_function_kw)
    run_and_check(compiled_call_func, [MISS, MISS, MISS, MISS], 1, caplog,
                  result, a, b, c)
    run_and_check(compiled_call_func, [HIT], 1, caplog, result, a, b, c)


class WithInner(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward_inner(self, x):
        return self.linear(x)

    def forward(self, x):

        def inner_forward(x):
            return self.forward_inner(x)

        return inner_forward(x)


def test_call_with_inner(caplog):
    reset()
    model = WithInner()
    a = torch.full((1, 1), 1.0)
    result = model(a)
    compiled_model = compile(model)
    run_and_check(compiled_model, [MISS, MISS, MISS], 1, caplog, result, a)
    run_and_check(compiled_model, [HIT], 1, caplog, result, a)


class WithNamedChildren(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)
        self.linear2 = torch.nn.Linear(1, 1)

    def forward(self, x):
        for name, layer in self.named_children():
            x = layer(x)
        return x


def test_call_with_named_children(caplog):
    reset()
    model = WithNamedChildren()
    a = torch.full((1, 1), 1.0)
    result = model(a)
    compiled_model = compile(model)
    run_and_check(compiled_model, [MISS], 1, caplog, result, a)
    run_and_check(compiled_model, [HIT], 1, caplog, result, a)


class WithListLayers(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.layers = []
        for i in range(3):
            layer = torch.nn.Linear(1, 1)
            layer_name = f'layer{i+1}'
            self.add_module(layer_name, layer)
            self.layers.append(layer_name)

    def forward(self, x):
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
        return x


def test_call_with_list_layers(caplog):
    reset()
    model = WithListLayers()
    a = torch.full((1, 1), 1.0)
    result = model(a)
    compiled_model = compile(model)
    run_and_check(compiled_model, [MISS], 1, caplog, result, a)
    run_and_check(compiled_model, [HIT], 1, caplog, result, a)


class AutogradUDFStatic(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, y):
        ctx.x = x
        return ctx.x + y


class AutogradUDFClass(torch.autograd.Function):

    @classmethod
    def forward(cls, ctx, x, y):
        ctx.x = x
        return ctx.x + y


def run_udf_static(x, y):
    return AutogradUDFStatic().apply(x, y)


def run_udf_class(x, y):
    return AutogradUDFClass().apply(x, y)


def test_udf_static(caplog):
    reset()
    x = 1.0
    y = torch.full((1, 1), 1.0)
    result = run_udf_static(x, y)
    compiled_model = compile(run_udf_static)
    run_and_check(compiled_model, [MISS, MISS, MISS], 1, caplog, result, x, y)
    run_and_check(compiled_model, [HIT], 1, caplog, result, x, y)


def test_udf_class(caplog):
    reset()
    with torch.no_grad():
        x = 1.0
        y = torch.full((1, 1), 1.0)
        result = run_udf_class(x, y)
        compiled_model = compile(run_udf_class)
        run_and_check(compiled_model, [MISS, MISS, MISS], 1, caplog, result, x,
                      y)
        run_and_check(compiled_model, [HIT], 1, caplog, result, x, y)