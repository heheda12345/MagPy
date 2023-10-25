import torch
from common.checker import run_and_check, HIT, MISS
from frontend.compile import compile, reset


def view_operation(a):
    shape = (2, 3)
    b = a.view(*shape)
    return b


def reshape_operation(a):
    shape = (2, 3)
    b = a.reshape(*shape)
    return b


def test_call_function_ex(caplog):
    reset()
    compiled2 = compile(view_operation)
    compiled3 = compile(reshape_operation)
    tensor = torch.tensor([1, 2, 3, 4, 5, 6])
    result = view_operation(tensor)
    run_and_check(compiled2, [MISS], 1, caplog, result, tensor)
    run_and_check(compiled2, [HIT], 1, caplog, result, tensor)
    result = reshape_operation(tensor)
    run_and_check(compiled3, [MISS], 2, caplog, result, tensor)
    run_and_check(compiled3, [HIT], 2, caplog, result, tensor)
