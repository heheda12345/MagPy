from frontend.compile import compile, reset
from common.checker import run_and_check, HIT, MISS
import torch

# def without_tensor_0(a):
#     return a

# def without_tensor_1(a):
#     return a * 3

# def without_tensor_2(a):
#     return (a, 'g')

# def without_tensor_3(a):
#     return a[1:]

# def without_tensor_4(a, b):
#     return a + b

# def test_without_tensor(caplog):
#     reset()
#     compiled_no_tensor0 = compile(without_tensor_0)
#     compiled_no_tensor1 = compile(without_tensor_1)
#     compiled_no_tensor2 = compile(without_tensor_2)
#     compiled_no_tensor3 = compile(without_tensor_3)
#     compiled_no_tensor4 = compile(without_tensor_4)
#     a = (1, 2.5, "abc")
#     b = (2, 4, "def")
#     result = without_tensor_0(a)
#     run_and_check(compiled_no_tensor0, [MISS], 1, caplog, result, a)
#     run_and_check(compiled_no_tensor0, [HIT], 1, caplog, result, a)
#     a = (10, 6, "j")
#     b = (8, 7, "5")
#     result = without_tensor_0(a)
#     run_and_check(compiled_no_tensor0, [MISS], 2, caplog, result, a)
#     run_and_check(compiled_no_tensor0, [HIT], 2, caplog, result, a)


def tensor_0():
    a = torch.full((1,), 5.0)
    b = torch.full((1,), 7.0)
    tuple_a = (1, 2, 4, a)
    tuple_b = (3.5, 7, b)
    return tuple_a[3] + tuple_b[2]


def tensor_1():
    a = torch.full((1,), 5.0)
    b = torch.full((1,), 7.0)
    tuple_a = (1, 2, 4, a)
    tuple_b = (3.5, 7, b)
    return tuple_a[3] + tuple_b[2]


def test_with_tensor(caplog):
    reset()
    compiled_tensor0 = compile(tensor_0)
    compiled_tensor1 = compile(tensor_1)
    # a = torch.full((1,), 5.0)
    # b = torch.full((1,), 7.0)
    # tuple_a = (1, 2, 4, a)
    # tuple_b = (3.5, 7, b)
    # result = tensor_0()
    # run_and_check(compiled_tensor0, [MISS], 1, caplog, result)
    # run_and_check(compiled_tensor0, [HIT], 1, caplog, result)
    # result = tensor_1()
    # run_and_check(compiled_tensor1, [MISS], 2, caplog, result)
    # run_and_check(compiled_tensor1, [HIT], 2, caplog, result)
