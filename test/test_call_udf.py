from frontend.compile import compile, reset
from common.checker import run_and_check, HIT, MISS
import torch


class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


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


def test_call_ud_func(caplog):
    reset()
    compiled_call_func = compile(call_func)
    a = torch.full((1, 1), 1.0)
    result = call_func(a)
    run_and_check(compiled_call_func, [MISS, MISS], 3, caplog, result, a)
    run_and_check(compiled_call_func, [HIT, HIT, HIT], 3, caplog, result, a)


def test_call_ud_model(caplog):
    reset()
    model = Model()
    a = torch.full((1, 1), 1.0)
    result = call_model(model, a)
    compiled_call_model = compile(call_model)
    run_and_check(compiled_call_model, [MISS, MISS], 3, caplog, result, model,
                  a)
    run_and_check(compiled_call_model, [HIT, HIT, HIT], 3, caplog, result,
                  model, a)


def test_call_ud_model2(caplog):
    reset()
    model = Model2()
    a = torch.full((1, 1), 1.0)
    result = model(a)
    compiled_model = compile(model)
    run_and_check(compiled_model, [MISS, MISS], 3, caplog, result, a)
    run_and_check(compiled_model, [HIT, HIT, HIT], 3, caplog, result, a)