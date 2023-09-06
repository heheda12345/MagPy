from frontend.compile import compile, reset
from common.checker import run_and_check, HIT, MISS
import torch


def store_no_break(a, b):
    c = a + b
    return c


def store_with_break(a, b):
    c = a + b
    d = c + a
    e = c + d
    f = c / 2 + e
    return f


def test_store(caplog):
    reset()
    compiled_store_no_break = compile(store_no_break)
    compiled_store_with_break = compile(store_with_break)
    a = torch.full((1,), 1.0)
    b = torch.full((1,), 2.0)
    result = store_no_break(a, b)
    run_and_check(compiled_store_no_break, [MISS], 1, caplog, result, a, b)
    run_and_check(compiled_store_no_break, [HIT], 1, caplog, result, a, b)
    result = store_with_break(a, b)
    run_and_check(compiled_store_with_break, [MISS], 3, caplog, result, a, b)
    run_and_check(compiled_store_with_break, [HIT, HIT], 3, caplog, result, a,
                  b)
