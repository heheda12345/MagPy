from frontend.compile import compile, reset
from common.checker import run_and_check, HIT, MISS, assert_equal
import torch


def without_tensor_0(a):
    return a


def without_tensor_1(a):
    return a * 3


def without_tensor_2(a):
    return [a, 9]


def without_tensor_3(a):
    return a[1:]


def without_tensor_4(a, b):
    return a + b


def test_without_tensor(caplog):
    reset()
    compiled_no_tensor0 = compile(without_tensor_0)
    compiled_no_tensor1 = compile(without_tensor_1)
    compiled_no_tensor2 = compile(without_tensor_2)
    compiled_no_tensor3 = compile(without_tensor_3)
    compiled_no_tensor4 = compile(without_tensor_4)
    a = [1, 2.5]
    b = [2, 4]
    result = without_tensor_0(a)
    run_and_check(compiled_no_tensor0, [MISS], 1, caplog, result, a)
    run_and_check(compiled_no_tensor0, [HIT], 1, caplog, result, a)
    result = without_tensor_1(a)
    run_and_check(compiled_no_tensor1, [MISS], 2, caplog, result, a)
    run_and_check(compiled_no_tensor1, [HIT], 2, caplog, result, a)
    result = without_tensor_2(a)
    run_and_check(compiled_no_tensor2, [MISS], 3, caplog, result, a)
    run_and_check(compiled_no_tensor2, [HIT], 3, caplog, result, a)
    result = without_tensor_3(a)
    run_and_check(compiled_no_tensor3, [MISS], 4, caplog, result, a)
    run_and_check(compiled_no_tensor3, [HIT], 4, caplog, result, a)
    result = without_tensor_4(a, b)
    run_and_check(compiled_no_tensor4, [MISS], 5, caplog, result, a, b)
    run_and_check(compiled_no_tensor4, [HIT], 5, caplog, result, a, b)
    a = [10, 6]
    b = [8, 7]
    result = without_tensor_0(a)
    run_and_check(compiled_no_tensor0, [MISS], 6, caplog, result, a)
    run_and_check(compiled_no_tensor0, [HIT], 6, caplog, result, a)
    result = without_tensor_4(a, b)
    run_and_check(compiled_no_tensor4, [MISS], 7, caplog, result, a, b)
    run_and_check(compiled_no_tensor4, [HIT], 7, caplog, result, a, b)


def tensor_0(list_a, list_b):
    return list_a[3] + list_b[2]


def tensor_1(list_a, list_b):
    return list_a[3] * list_b[2]


def tensor_2(list_a):
    return list_a


def tensor_3(list_a, list_b):
    return list_a + list_b


def tensor_4(list_a, list_b):
    return list_a + [3]


def list_id(list_a, list_b):
    c = list_a + list_b
    return c[3], c[6]


def test_with_tensor(caplog):
    reset()
    compiled_tensor0 = compile(tensor_0)
    compiled_tensor1 = compile(tensor_1)
    compiled_tensor2 = compile(tensor_2)
    compiled_tensor3 = compile(tensor_3)
    compiled_tensor4 = compile(tensor_4)
    compiled_tensor5 = compile(list_id)
    a = torch.full((1,), 5.0)
    b = torch.full((1,), 7.0)
    list_a = [1, 2, 4, a]
    list_b = [3.5, 7, b]
    result = tensor_0(list_a, list_b)
    run_and_check(compiled_tensor0, [MISS], 1, caplog, result, list_a, list_b)
    run_and_check(compiled_tensor0, [HIT], 1, caplog, result, list_a, list_b)
    result = tensor_1(list_a, list_b)
    run_and_check(compiled_tensor1, [MISS], 2, caplog, result, list_a, list_b)
    run_and_check(compiled_tensor1, [HIT], 2, caplog, result, list_a, list_b)
    result = tensor_2(list_a)
    run_and_check(compiled_tensor2, [MISS], 3, caplog, result, list_a)
    run_and_check(compiled_tensor2, [HIT], 3, caplog, result, list_a)
    result = tensor_3(list_a, list_b)
    run_and_check(compiled_tensor3, [MISS], 4, caplog, result, list_a, list_b)
    run_and_check(compiled_tensor3, [HIT], 4, caplog, result, list_a, list_b)
    result = tensor_4(list_a, list_b)
    run_and_check(compiled_tensor4, [MISS], 5, caplog, result, list_a, list_b)
    run_and_check(compiled_tensor4, [HIT], 5, caplog, result, list_a, list_b)
    list_a = [1, 2, 4, a]
    list_b = [3.5, 7, a]
    result = list_id(list_a, list_b)
    assert_equal(id(result[0]), id(result[1]))
    assert_equal(id(result[0]), id(compiled_tensor5(list_a, list_b)[1]))
    assert_equal(id(compiled_tensor5(list_a, list_b)[0]),
                 id(compiled_tensor5(list_a, list_b)[1]))
    # test nested list
    list_a = [1, 2, 4, (6, 7), a, [8, (9, 10), 11]]
    list_b = [3.5, 7, b]
    result = tensor_3(list_a, list_b)
    run_and_check(compiled_tensor3, [MISS], 7, caplog, result, list_a, list_b)
    run_and_check(compiled_tensor3, [HIT], 7, caplog, result, list_a, list_b)
