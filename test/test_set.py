from frontend.compile import compile, reset
from common.checker import run_and_check, HIT, MISS, assert_equal
import torch


def without_tensor_0(a):
    return a


def without_tensor_1(a, b):
    return a - b


def without_tensor_2(a, b):
    return a | b


def without_tensor_3(a, b):
    return a & b


def without_tensor_4(a, b):
    return a ^ b


def test_without_tensor(caplog):
    reset()
    compiled_no_tensor0 = compile(without_tensor_0)
    compiled_no_tensor1 = compile(without_tensor_1)
    compiled_no_tensor2 = compile(without_tensor_2)
    compiled_no_tensor3 = compile(without_tensor_3)
    compiled_no_tensor4 = compile(without_tensor_4)
    a = {2.8, 2.1, 2.2, 2.1}
    b = {2.8, 3.9, 4.1}
    result = without_tensor_0(a)
    run_and_check(compiled_no_tensor0, [MISS], 1, caplog, result, a)
    run_and_check(compiled_no_tensor0, [HIT], 1, caplog, result, a)
    result = without_tensor_1(a, b)
    run_and_check(compiled_no_tensor1, [MISS], 2, caplog, result, a, b)
    run_and_check(compiled_no_tensor1, [HIT], 2, caplog, result, a, b)
    result = without_tensor_2(a, b)
    run_and_check(compiled_no_tensor2, [MISS], 3, caplog, result, a, b)
    run_and_check(compiled_no_tensor2, [HIT], 3, caplog, result, a, b)
    result = without_tensor_3(a, b)
    run_and_check(compiled_no_tensor3, [MISS], 4, caplog, result, a, b)
    run_and_check(compiled_no_tensor3, [HIT], 4, caplog, result, a, b)
    result = without_tensor_4(a, b)
    run_and_check(compiled_no_tensor4, [MISS], 5, caplog, result, a, b)
    run_and_check(compiled_no_tensor4, [HIT], 5, caplog, result, a, b)
    a = {10.1, 6.2}
    b = {8.4, 7.2}
    result = without_tensor_0(a)
    run_and_check(compiled_no_tensor0, [MISS], 6, caplog, result, a)
    run_and_check(compiled_no_tensor0, [HIT], 6, caplog, result, a)
    result = without_tensor_4(a, b)
    run_and_check(compiled_no_tensor4, [MISS], 7, caplog, result, a, b)
    run_and_check(compiled_no_tensor4, [HIT], 7, caplog, result, a, b)


def tensor_0(a, b):
    return {1, 2, 3, a + b}


def test_with_tensor(caplog):
    reset()
    compiled_tensor0 = compile(without_tensor_0)
    compiled_tensor1 = compile(without_tensor_1)
    compiled_tensor2 = compile(without_tensor_2)
    compiled_tensor3 = compile(without_tensor_3)
    compiled_tensor4 = compile(without_tensor_4)
    compiled_tensor5 = compile(tensor_0)
    a = torch.full((1,), 5.0)
    b = torch.full((1,), 7.0)
    c = torch.full((1,), 7.0)
    set_a = {1, 2, 4, a, 4, 1}
    set_b = {3.5, 7, b, 4, 2, c}
    result = without_tensor_0(set_a)
    run_and_check(compiled_tensor0, [MISS], 1, caplog, result, set_a)
    run_and_check(compiled_tensor0, [HIT], 1, caplog, result, set_a)
    result = without_tensor_1(set_a, set_b)
    run_and_check(compiled_tensor1, [MISS], 2, caplog, result, set_a, set_b)
    run_and_check(compiled_tensor1, [HIT], 2, caplog, result, set_a, set_b)
    result = without_tensor_2(set_a, set_b)
    run_and_check(compiled_tensor2, [MISS], 3, caplog, result, set_a, set_b)
    run_and_check(compiled_tensor2, [HIT], 3, caplog, result, set_a, set_b)
    result = without_tensor_3(set_a, set_b)
    run_and_check(compiled_tensor3, [MISS], 4, caplog, result, set_a, set_b)
    run_and_check(compiled_tensor3, [HIT], 4, caplog, result, set_a, set_b)
    result = without_tensor_4(set_a, set_b)
    run_and_check(compiled_tensor4, [MISS], 5, caplog, result, set_a, set_b)
    run_and_check(compiled_tensor4, [HIT], 5, caplog, result, set_a, set_b)
    # test nested set
    set_a = {1, 2, 4, (6, 7), a, (8, (9, 10), 11)}
    set_b = {3.5, 7, b, (6.6, 8.8)}
    result = without_tensor_3(set_a, set_b)
    run_and_check(compiled_tensor3, [MISS], 6, caplog, result, set_a, set_b)
    run_and_check(compiled_tensor3, [HIT], 6, caplog, result, set_a, set_b)
    result = tensor_0(a, b)
    run_and_check(compiled_tensor5, [MISS], 7, caplog, result, a, b)
    run_and_check(compiled_tensor5, [HIT], 7, caplog, result, a, b)
    a = torch.full((1,), 6.0)
    b = torch.full((1,), 7.0)
    result = tensor_0(a, b)
    run_and_check(compiled_tensor5, [HIT], 7, caplog, result, a, b)
