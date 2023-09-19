from frontend.compile import compile, reset
from common.checker import run_and_check, HIT, MISS, assert_equal
import torch


def without_tensor_0(a):
    return a


def without_tensor_1(a):
    return a * 3


def without_tensor_2(a):
    return (a, 9)


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
    a = (1, 2.5)
    b = (2, 4)
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
    a = (10, 6)
    b = (8, 7)
    result = without_tensor_0(a)
    run_and_check(compiled_no_tensor0, [MISS], 6, caplog, result, a)
    run_and_check(compiled_no_tensor0, [HIT], 6, caplog, result, a)
    result = without_tensor_4(a, b)
    run_and_check(compiled_no_tensor4, [MISS], 7, caplog, result, a, b)
    run_and_check(compiled_no_tensor4, [HIT], 7, caplog, result, a, b)


def tensor_0(tuple_a, tuple_b):
    return tuple_a[3] + tuple_b[2]


def tensor_1(tuple_a, tuple_b):
    return tuple_a[3] * tuple_b[2]


def tensor_2(tuple_a):
    return tuple_a


def tensor_3(tuple_a, tuple_b):
    return tuple_a + tuple_b


def tensor_4(tuple_a, tuple_b):
    return tuple_a + (3,)


def tuple_id(tuple_a, tuple_b):
    c = tuple_a + tuple_b
    return c[3], c[6]


def test_with_tensor(caplog):
    reset()
    compiled_tensor0 = compile(tensor_0)
    compiled_tensor1 = compile(tensor_1)
    compiled_tensor2 = compile(tensor_2)
    compiled_tensor3 = compile(tensor_3)
    compiled_tensor4 = compile(tensor_4)
    compiled_tensor5 = compile(tuple_id)
    a = torch.full((1,), 5.0)
    b = torch.full((1,), 7.0)
    tuple_a = (1, 2, 4, a)
    tuple_b = (3.5, 7, b)
    result = tensor_0(tuple_a, tuple_b)
    run_and_check(compiled_tensor0, [MISS], 1, caplog, result, tuple_a, tuple_b)
    run_and_check(compiled_tensor0, [HIT], 1, caplog, result, tuple_a, tuple_b)
    result = tensor_1(tuple_a, tuple_b)
    run_and_check(compiled_tensor1, [MISS], 2, caplog, result, tuple_a, tuple_b)
    run_and_check(compiled_tensor1, [HIT], 2, caplog, result, tuple_a, tuple_b)
    result = tensor_2(tuple_a)
    run_and_check(compiled_tensor2, [MISS], 3, caplog, result, tuple_a)
    run_and_check(compiled_tensor2, [HIT], 3, caplog, result, tuple_a)
    result = tensor_3(tuple_a, tuple_b)
    run_and_check(compiled_tensor3, [MISS], 4, caplog, result, tuple_a, tuple_b)
    run_and_check(compiled_tensor3, [HIT], 4, caplog, result, tuple_a, tuple_b)
    result = tensor_4(tuple_a, tuple_b)
    run_and_check(compiled_tensor4, [MISS], 5, caplog, result, tuple_a, tuple_b)
    run_and_check(compiled_tensor4, [HIT], 5, caplog, result, tuple_a, tuple_b)
    tuple_a = (1, 2, 4, a)
    tuple_b = (3.5, 7, a)
    result = tuple_id(tuple_a, tuple_b)
    assert_equal(id(result[0]), id(result[1]))
    assert_equal(id(result[0]), id(compiled_tensor5(tuple_a, tuple_b)[1]))
    assert_equal(id(compiled_tensor5(tuple_a, tuple_b)[0]),
                 id(compiled_tensor5(tuple_a, tuple_b)[1]))