from frontend.compile import compile, reset
from frontend.utils import add_force_graph_break
from frontend.c_api import get_next_frame_id
from common.checker import run_and_check, HIT, MISS, assert_equal
import torch


def inplace_add(a, b):
    a += b
    return a


def test_inplace_add(caplog):
    reset()
    compiled = compile(inplace_add)
    result1 = inplace_add(1.0, 2.0)
    run_and_check(compiled, [MISS], 1, caplog, result1, 1.0, 2.0)
    run_and_check(compiled, [HIT], 1, caplog, result1, 1.0, 2.0)

    result2 = inplace_add((1, 2), (3, 4))
    run_and_check(compiled, [MISS], 2, caplog, result2, (1, 2), (3, 4))
    run_and_check(compiled, [HIT], 2, caplog, result2, (1, 2), (3, 4))

    def get_input3():
        return [1, 2], [3, 4]

    result3 = inplace_add(*get_input3())
    run_and_check(compiled, [MISS], 3, caplog, result3, *get_input3())
    run_and_check(compiled, [HIT], 3, caplog, result3, *get_input3())
    input3 = get_input3()
    output3 = compiled(*input3)
    assert_equal(id(input3[0]), id(output3))

    result4 = inplace_add(torch.tensor(1), torch.tensor(2))
    run_and_check(compiled, [MISS], 4, caplog, result4, torch.tensor(1),
                  torch.tensor(2))
    result5 = inplace_add(torch.tensor(3), torch.tensor(4))
    run_and_check(compiled, [HIT], 4, caplog, result5, torch.tensor(3),
                  torch.tensor(4))
    input6 = (torch.tensor(5), torch.tensor(6))
    result6 = compiled(*input6)
    assert_equal(id(input6[0]), id(result6))


# TODO:
# def inplace_add2(a, b):
#     a += b
#     return b # but a is still modified


def store_subscr_add(a, b):
    a[1] += b
    return a


def test_inplace_subscr_add(caplog):
    reset()
    compiled = compile(store_subscr_add)

    def get_input1():
        return [1, 2], 3

    result1 = store_subscr_add(*get_input1())
    run_and_check(compiled, [MISS], 1, caplog, result1, *get_input1())
    run_and_check(compiled, [HIT], 1, caplog, result1, *get_input1())
    input1 = get_input1()
    output1 = compiled(*input1)
    assert_equal(id(input1[0]), id(output1))

    def get_input2():
        return torch.tensor([1, 2]), torch.tensor(3)

    result2 = store_subscr_add(*get_input2())
    run_and_check(compiled, [MISS], 2, caplog, result2, *get_input2())
    run_and_check(compiled, [HIT], 2, caplog, result2, *get_input2())
    input2 = get_input2()
    output2 = compiled(*input2)
    assert_equal(id(input2[0]), id(output2))


def store_subscr(a, b):
    a[1] = b
    return a, b


def test_store_subscr(caplog):
    reset()
    compiled = compile(store_subscr)

    def get_input1():
        return [1, 2], [3, 4]

    result = store_subscr(*get_input1())
    run_and_check(compiled, [MISS], 1, caplog, result, *get_input1())
    run_and_check(compiled, [HIT], 1, caplog, result, *get_input1())
    a, b = get_input1()
    output = compiled(a, b)
    assert_equal(id(a), id(output[0]))
    assert_equal(id(b), id(output[0][1]))
    assert_equal(id(b), id(output[1]))


def store_without_return(a, b):
    a[1] = b
    return b


def test_store_without_return(caplog):
    reset()
    compiled = compile(store_without_return)

    def get_input1():
        return [1, 2], [3, 4]

    a, b = get_input1()
    result = store_without_return(a, b)

    run_and_check(compiled, [MISS], 1, caplog, result, *get_input1())
    run_and_check(compiled, [HIT], 1, caplog, result, *get_input1())

    a1, b1 = get_input1()
    output = compiled(a1, b1)
    assert_equal(id(b1), id(output))
    assert_equal(a1, a)


def store_to_temp1(a):
    b = [1, 2, 3]
    b[2] = a
    return b


def store_to_temp2(a):
    b = [1, 2, 3]
    b[2] = a
    return a


def test_store_to_temp(caplog):
    reset()

    result = store_to_temp1(4)
    compiled = compile(store_to_temp1)
    run_and_check(compiled, [MISS], 1, caplog, result, 4)
    run_and_check(compiled, [HIT], 1, caplog, result, 4)

    result = store_to_temp2(4)
    compiled = compile(store_to_temp2)
    run_and_check(compiled, [MISS], 2, caplog, result, 4)
    run_and_check(compiled, [HIT], 2, caplog, result, 4)


def inplace_callee_no_ret(a):
    a[1] = 2.0


def inplace_callee_ret(a):
    a[1] = 2.0
    return a


def inplace_callee_add_no_ret(a):
    a[1] += 2.0


def inplace_callee_add_ret(a):
    a[1] += 2.0
    return a


def caller1(a):
    inplace_callee_no_ret(a)
    return a


def caller2(a):
    inplace_callee_no_ret(a)


def caller3(a):
    inplace_callee_ret(a)
    return a


def caller4(a):
    inplace_callee_ret(a)


def caller5(a):
    inplace_callee_add_no_ret(a)
    return a


def caller6(a):
    inplace_callee_add_no_ret(a)


def caller7(a):
    inplace_callee_add_ret(a)
    return a


def caller8(a):
    inplace_callee_add_ret(a)


def test_inplace_function(caplog):
    reset()
    fs = [
        caller1, caller2, caller3, caller4, caller5, caller6, caller7, caller8
    ]

    def get_input1():
        return [1.0, 3.0]

    def get_input2():
        return torch.tensor([1.0, 3.0])

    cache_size = 0
    for f in fs:
        print("===============running", f)
        compiled = compile(f)
        cache_size += 1
        original_input = get_input1()
        result = f(original_input)
        run_and_check(compiled, [MISS, MISS], cache_size, caplog, result,
                      get_input1())
        run_and_check(compiled, [HIT], cache_size, caplog, result, get_input1())
        input1 = get_input1()
        _output = compiled(input1)
        assert_equal(input1, original_input)

        compiled = compile(f)
        cache_size += 1
        original_input = get_input2()
        result = f(original_input)
        run_and_check(compiled, [MISS, MISS], cache_size, caplog, result,
                      get_input2())
        run_and_check(compiled, [HIT], cache_size, caplog, result, get_input2())
        input2 = get_input2()
        _output = compiled(input2)
        assert_equal(input2, original_input)