from frontend.compile import compile, reset
from common.checker import run_and_check, HIT, MISS, assert_equal
import torch


def without_tensor_0(a):
    return a


def test_without_tensor(caplog):
    reset()
    compiled_no_tensor0 = compile(without_tensor_0)
    a = {2.8: 1.1, 2.1: 2, 2.2: 3.3}
    result = without_tensor_0(a)
    run_and_check(compiled_no_tensor0, [MISS], 1, caplog, result, a)
    run_and_check(compiled_no_tensor0, [HIT], 1, caplog, result, a)


def tensor_0(a, b):
    return {1: 1, 2: 4, 3: a + b}


def test_with_tensor(caplog):
    reset()
    compiled_tensor0 = compile(without_tensor_0)
    compiled_tensor5 = compile(tensor_0)
    a = torch.full((1,), 5.0)
    b = torch.full((1,), 7.0)
    c = torch.full((1,), 7.0)
    dict_a = {1: c, 2: b, 4: 1, 3: a}
    result = without_tensor_0(dict_a)
    run_and_check(compiled_tensor0, [MISS], 1, caplog, result, dict_a)
    run_and_check(compiled_tensor0, [HIT], 1, caplog, result, dict_a)
    result = tensor_0(a, b)
    run_and_check(compiled_tensor5, [MISS], 2, caplog, result, a, b)
    run_and_check(compiled_tensor5, [HIT], 2, caplog, result, a, b)
    a = torch.full((1,), 6.0)
    b = torch.full((1,), 7.0)
    result = tensor_0(a, b)
    run_and_check(compiled_tensor5, [HIT], 2, caplog, result, a, b)
    # test nested dict
    dict_a = {1: [a, b, c], 2: (b, 2.2), 4: 1, 3: a}
    result = without_tensor_0(dict_a)
    run_and_check(compiled_tensor0, [MISS], 3, caplog, result, dict_a)
    run_and_check(compiled_tensor0, [HIT], 3, caplog, result, dict_a)