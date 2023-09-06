from frontend.compile import compile, reset
from common.checker import run_and_check, HIT, MISS
import torch


def tensor_only(a, b, c):
    return a + b + c


def tensor_add_float(a, b):
    return a + b + 1.0


def tensor_add_int(a, b):
    return a + b + 1


def test_tensor_only(caplog):
    reset()
    compiled_tensor_only = compile(tensor_only)
    a = torch.full((1,), 1.0)
    b = torch.full((1,), 2.0)
    c = torch.full((1,), 3.0)
    result = tensor_only(a, b, c)
    run_and_check(compiled_tensor_only, [MISS], 1, caplog, result, a, b, c)
    run_and_check(compiled_tensor_only, [HIT], 1, caplog, result, a, b, c)
    a = torch.full((2,), 1.0)
    b = torch.full((2,), 2.0)
    c = torch.full((2,), 3.0)
    result = tensor_only(a, b, c)
    run_and_check(compiled_tensor_only, [MISS], 2, caplog, result, a, b, c)
    run_and_check(compiled_tensor_only, [HIT], 2, caplog, result, a, b, c)


def test_with_scalar(caplog):
    reset()
    compiled_tensor_add_float = compile(tensor_add_float)
    compiled_tensor_add_int = compile(tensor_add_int)
    a = torch.full((1,), 1.0)
    b = torch.full((1,), 2.0)
    result = tensor_add_float(a, b)
    run_and_check(compiled_tensor_add_float, [MISS], 1, caplog, result, a, b)
    run_and_check(compiled_tensor_add_float, [HIT], 1, caplog, result, a, b)
    result = tensor_add_int(a, b)
    run_and_check(compiled_tensor_add_int, [MISS], 2, caplog, result, a, b)
    run_and_check(compiled_tensor_add_int, [HIT], 2, caplog, result, a, b)