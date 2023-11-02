import pytest
from frontend.compile import compile, reset
from frontend.utils import add_force_graph_break
from frontend.c_api import get_next_frame_id
import logging
from common.checker import run_and_check, HIT, MISS
import torch


def perfect0(a):
    return a + 1


def perfect1(a):
    return a + 1


def graph_break0(a):
    return (a + 1) // 2


def graph_break1(a):
    return (a + 1) // 2 + 1


def test_perfect(caplog):
    reset()
    compiled_perfect0 = compile(perfect0)
    compiled_perfect1 = compile(perfect1)
    run_and_check(compiled_perfect0, [MISS], 1, caplog, 4, 3)
    run_and_check(compiled_perfect0, [HIT], 1, caplog, 4, 3)
    run_and_check(compiled_perfect0, [MISS], 2, caplog, 5, 4)
    run_and_check(compiled_perfect0, [HIT], 2, caplog, 5, 4)
    run_and_check(compiled_perfect1, [MISS], 3, caplog, 4, 3)
    run_and_check(compiled_perfect1, [HIT], 3, caplog, 4, 3)
    run_and_check(compiled_perfect1, [MISS], 4, caplog, 5, 4)
    run_and_check(compiled_perfect1, [HIT], 4, caplog, 5, 4)


def test_graph_break(caplog):
    reset()
    compiled_graph_break0 = compile(graph_break0)
    add_force_graph_break(get_next_frame_id(), 4)
    run_and_check(compiled_graph_break0, [MISS], 1, caplog, 2, 3)
    run_and_check(compiled_graph_break0, [HIT], 1, caplog, 2, 3)
    run_and_check(compiled_graph_break0, [MISS], 2, caplog, 2, 4)
    run_and_check(compiled_graph_break0, [HIT], 2, caplog, 2, 4)

    compiled_graph_break1 = compile(graph_break1)
    add_force_graph_break(get_next_frame_id(), 4)
    run_and_check(compiled_graph_break1, [MISS], 4, caplog, 2, 1)
    run_and_check(compiled_graph_break1, [HIT, HIT], 4, caplog, 2, 1)
    run_and_check(compiled_graph_break1, [MISS, HIT], 5, caplog, 2, 2)
    run_and_check(compiled_graph_break1, [HIT, HIT], 5, caplog, 2, 2)


def perfect0_float(a):
    return a + 1.0


def perfect1_float(a):
    return a + 1.0


def graph_break0_float(a):
    return (a + 1.0) / 2


def graph_break1_float(a):
    return (a + 1.0) / 2 + 1


def test_perfect_float(caplog):
    reset()
    compiled_perfect0 = compile(perfect0_float)
    compiled_perfect1 = compile(perfect1_float)
    run_and_check(compiled_perfect0, [MISS], 1, caplog, 4.0, 3.0)
    run_and_check(compiled_perfect0, [HIT], 1, caplog, 4.0, 3.0)
    run_and_check(compiled_perfect0, [MISS], 2, caplog, 5.0, 4.0)
    run_and_check(compiled_perfect0, [HIT], 2, caplog, 5.0, 4.0)
    run_and_check(compiled_perfect1, [MISS], 3, caplog, 4.0, 3.0)
    run_and_check(compiled_perfect1, [HIT], 3, caplog, 4.0, 3.0)
    run_and_check(compiled_perfect1, [MISS], 4, caplog, 5.0, 4.0)
    run_and_check(compiled_perfect1, [HIT], 4, caplog, 5.0, 4.0)


def test_graph_break_float(caplog):
    reset()
    add_force_graph_break(get_next_frame_id(), 4)
    compiled_graph_break0 = compile(graph_break0_float)
    run_and_check(compiled_graph_break0, [MISS], 1, caplog, 2.0, 3.0)
    run_and_check(compiled_graph_break0, [HIT], 1, caplog, 2.0, 3.0)
    run_and_check(compiled_graph_break0, [MISS], 2, caplog, 2.5, 4.0)
    run_and_check(compiled_graph_break0, [HIT], 2, caplog, 2.5, 4.0)

    add_force_graph_break(get_next_frame_id(), 4)
    compiled_graph_break1 = compile(graph_break1_float)
    run_and_check(compiled_graph_break1, [MISS], 4, caplog, 2.0, 1.0)
    run_and_check(compiled_graph_break1, [HIT, HIT], 4, caplog, 2.0, 1.0)


def binary_add(a, b):
    return a + b


def binary_subtract(a, b):
    return a - b


def binary_multiply(a, b):
    return a * b


def binary_floor_divide(a, b):
    return a // b


def binary_true_divide(a, b):
    return a / b


def binary_mod(a, b):
    return a % b


def binary_power(a, b):
    return a**b


def binary_lshift(a, b):
    return a << b


def binary_rshift(a, b):
    return a >> b


def binary_and(a, b):
    return a & b


def binary_xor(a, b):
    return a ^ b


def binary_or(a, b):
    return a | b


def test_binary_op(caplog):
    reset()
    funcs = [
        binary_add, binary_subtract, binary_multiply, binary_floor_divide,
        binary_true_divide, binary_mod, binary_power, binary_lshift,
        binary_rshift, binary_and, binary_xor, binary_or
    ]
    compiled_funcs = [compile(func) for func in funcs]
    cache_cnt = 0
    for func, compiled_func in zip(funcs, compiled_funcs):
        for a in [1, 2, 3]:
            for b in [1, 2, 3]:
                cache_cnt += 1
                run_and_check(compiled_func, [MISS], cache_cnt, caplog,
                              func(a, b), a, b)
    for func, compiled_func in zip(funcs, compiled_funcs):
        for a in [1, 2, 3]:
            for b in [1, 2, 3]:
                run_and_check(compiled_func, [HIT], cache_cnt, caplog,
                              func(a, b), a, b)


def add3(a, b, c):
    return a + b + c


def call_add3(a, b, c):
    return add3(a, b, c)


def test_dyn_int(caplog):
    reset()
    import frontend.dynamic as dyn
    a = 1
    b = 2
    c = 3
    dyn.mark_dynamic(a, dyn.ScalarWithUnknownValue())
    dyn.mark_dynamic(b, dyn.ScalarWithUnknownValue())
    compiled_fn = compile(add3)
    run_and_check(compiled_fn, [MISS], 1, caplog, 6, a, b, c)
    run_and_check(compiled_fn, [HIT], 1, caplog, 6, a, b, c)
    run_and_check(compiled_fn, [HIT], 1, caplog, 12, 4, 5, 3)
    run_and_check(compiled_fn, [MISS], 2, caplog, 8, a, b, 5)


def test_dyn_int_with_call(caplog):
    reset()
    import frontend.dynamic as dyn
    a = 1
    b = 2
    c = 3
    dyn.mark_dynamic(a, dyn.ScalarWithUnknownValue())
    dyn.mark_dynamic(b, dyn.ScalarWithUnknownValue())
    compiled_fn = compile(call_add3)
    run_and_check(compiled_fn, [MISS, MISS], 1, caplog, 6, a, b, c)
    run_and_check(compiled_fn, [HIT], 1, caplog, 6, a, b, c)
    run_and_check(compiled_fn, [HIT], 1, caplog, 12, 4, 5, 3)
    run_and_check(compiled_fn, [MISS, MISS], 2, caplog, 8, a, b, 5)


def dynamic_scalar_from_tensor(a, b, c):
    d = float(a + b)
    return c + d


def test_dynamic_scalar_from_tensor(caplog):
    reset()
    a = torch.tensor(1.0)
    b = torch.tensor(2.0)
    c = 3.0
    expect = dynamic_scalar_from_tensor(a, b, c)
    compiled = compile(dynamic_scalar_from_tensor)
    run_and_check(compiled, [MISS], 1, caplog, expect, a, b, c)
    run_and_check(compiled, [HIT], 1, caplog, expect, a, b, c)
    aa = torch.tensor(4.0)
    bb = torch.tensor(5.0)
    expect = dynamic_scalar_from_tensor(aa, bb, c)
    run_and_check(compiled, [HIT], 1, caplog, expect, aa, bb, c)
