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
