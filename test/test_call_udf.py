from frontend.compile import compile, reset
from common.checker import run_and_check, HIT, MISS
import torch


def g(a):
    return a + 1


def f(a):
    b = g(a)
    return b


def test_call_udf_func(caplog):
    reset()
    compiled_f = compile(f)
    a = torch.tensor(1.0)
    result = f(a)
    run_and_check(compiled_f, [MISS, MISS], 3, caplog, result, a)
    run_and_check(compiled_f, [HIT, HIT, HIT], 3, caplog, result, a)
