from frontend.compile import compile, reset
from frontend.utils import add_force_graph_break
from frontend.c_api import get_next_frame_id
from common.checker import run_and_check, HIT, MISS, ALL_MISS
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


def func_kw_with_self(a):
    return a.clamp(min=1e-8)


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


def test_call_kw_with_self(caplog):
    reset()
    compiled_call_func = compile(func_kw_with_self)
    a = torch.tensor(1.0)
    result = func_kw_with_self(a)
    run_and_check(compiled_call_func, [MISS], 1, caplog, result, a)
    run_and_check(compiled_call_func, [HIT], 1, caplog, result, a)


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


class AutogradUDFEmpty(torch.autograd.Function):

    @classmethod
    def forward(cls, ctx, x):
        return x

    @classmethod
    def backward(cls, ctx, grad):
        return grad


def run_udf_static(x, y):
    return AutogradUDFStatic().apply(x, y)


def run_udf_class(x, y):
    return AutogradUDFClass().apply(x, y)


def call_run_udf(x):
    return AutogradUDFEmpty().apply(x)


def test_udf_static(caplog):
    reset()
    x = 1.0
    y = torch.full((1, 1), 1.0)
    result = run_udf_static(x, y)
    compiled_model = compile(run_udf_static)
    run_and_check(compiled_model, [MISS, MISS], 1, caplog, result, x, y)
    run_and_check(compiled_model, [HIT], 1, caplog, result, x, y)


def test_udf_class(caplog):
    reset()
    with torch.no_grad():
        x = 1.0
        y = torch.full((1, 1), 1.0)
        result = run_udf_class(x, y)
        compiled_model = compile(run_udf_class)
        run_and_check(compiled_model, [MISS, MISS], 1, caplog, result, x, y)
        run_and_check(compiled_model, [HIT], 1, caplog, result, x, y)


def test_call_run_udf(caplog):
    reset()
    with torch.no_grad():
        # x = 1.0
        x = torch.full((1, 1), 1.0)
        result = call_run_udf(x)
        compiled_model = compile(call_run_udf)
        run_and_check(compiled_model, [MISS, MISS], 1, caplog, result, x)
        run_and_check(compiled_model, [HIT], 1, caplog, result, x)


def b_aba(a):
    return a_aba(a + 2.0, 0)


def a_aba(a, b):
    if b == 0:
        return a + 3.0
    else:
        return b_aba(a + 1.0)


def test_call_aba(caplog):
    reset()
    compiled = compile(a_aba)
    expect = a_aba(1.0, 1)
    run_and_check(compiled, [MISS, MISS, MISS], 1, caplog, expect, 1.0, 1)
    run_and_check(compiled, [HIT], 1, caplog, expect, 1.0, 1)


def list_zip1(x):
    y = list(zip(x))
    return y[0] + y[1] + y[2]


def list_zip2(x):
    y = list(zip(x))
    return y[0][0] + y[1][0] + y[2][0]


def map_fn(x):
    return x + 1


def map_caller1(x):
    lx = []
    mp = map(map_fn, lx)
    lst = list(mp)
    return lst


def map_caller2(x):
    lx = [x, x]
    mp = map(map_fn, lx)
    lst = list(mp)
    return lst


def map_caller3(x):
    lx = x
    mp = map(map_fn, lx)
    lst = list(mp)
    return lst


def test_call_high_order(caplog):
    reset()
    for i, func in enumerate(
        (list_zip1,)):  #, list_zip2, map_caller1, map_caller2, map_caller3)):
        x = torch.rand((3, 3))
        expect = func(x)
        compiled = compile(func)
        run_and_check(compiled, [ALL_MISS], i + 1, caplog, expect, x)
        run_and_check(compiled, [HIT], i + 1, caplog, expect, x)


def func_attr1(a):
    b = a + 2.0
    return b


def func_attr2(c):
    d = c.data + 3.0
    return d


def func_attr(e):
    e1 = e + 1.0
    out = func_attr1(e1)
    return func_attr2(out)


def test_guard_attr(caplog):
    reset()
    para = torch.full((1,), 1.0)
    compiled = compile(func_attr)
    expect = func_attr(para)
    run_and_check(compiled, [MISS, MISS, MISS], 1, caplog, expect, para)
    run_and_check(compiled, [HIT], 1, caplog, expect, para)


def f(b):

    def g():
        return a + 1

    a = b + 1
    return g()


def test_empty_cell(caplog):
    reset()
    compiled = compile(f)
    expect = f(1)
    run_and_check(compiled, [MISS, MISS], 1, caplog, expect, 1)
    run_and_check(compiled, [HIT], 1, caplog, expect, 1)
    x = torch.rand((3, 3))
    expect = f(x)
    run_and_check(compiled, [MISS, MISS], 2, caplog, expect, x)
    run_and_check(compiled, [HIT], 2, caplog, expect, x)


class parent_func_call(torch.nn.Sequential):

    def forward(self):
        return self.parameters()


def test_parent_func_call(caplog):
    reset()
    instance = parent_func_call()
    expect = instance()
    compiled = compile(instance)
    run_and_check(compiled, [MISS], 1, caplog, expect)
    run_and_check(compiled, [HIT], 1, caplog, expect)


def para_with_star(a, *b):
    out = []
    for i in b:
        out.append(i)
    return out

def call_with_star(a):
    intput1 = 3
    b = a
    out = para_with_star(intput1, b, 6.2)
    return out


def para_with_tuple(a, input2):
    out = []
    for i in input2:
        out.append(i)
    return out

def call_with_tuple(a):
    intput1 = 3
    out = para_with_tuple(intput1, (a, 5))
    return out

def test_call_parameter(caplog):
    reset()
    compiled1 = compile(call_with_star)
    compiled2 = compile(call_with_tuple)
    input = torch.randn([2, 2])
    expect = call_with_star(input)
    run_and_check(compiled1, [ALL_MISS], 1, caplog, expect, input)
    run_and_check(compiled1, [HIT], 1, caplog, expect, input)
    expect = call_with_tuple(input)
    run_and_check(compiled2, [ALL_MISS], 2, caplog, expect, input)
    run_and_check(compiled2, [HIT], 2, caplog, expect, input)