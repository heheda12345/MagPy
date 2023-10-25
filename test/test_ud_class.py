import torch
from frontend.compile import compile, reset
from common.checker import run_and_check, HIT, MISS, assert_equal, run_and_check_cache


class A:

    def __init__(self, x) -> None:
        self.x = x


def assign_to_exist(a):
    a.x = a.x + 1


def test_assign_to_exist(caplog):
    reset()
    x = torch.randn(3, 4)
    a1 = A(x)
    a2 = A(x)
    a3 = A(x)
    expect = assign_to_exist(a1)
    compiled = compile(assign_to_exist)
    run_and_check(compiled, [MISS], 1, caplog, expect, a2)
    run_and_check(compiled, [HIT], 1, caplog, expect, a3)
    assert_equal(a2.x, a1.x)
    assert_equal(a3.x, a1.x)


def create_new_class(x):
    a = A(x + 1)
    return a


def test_create_new_class(caplog):
    reset()
    x = torch.randn(4)
    y = create_new_class(x)
    compiled = compile(create_new_class)
    run_and_check_cache(compiled, [MISS, MISS], 1, caplog, x)
    run_and_check_cache(compiled, [HIT], 1, caplog, x)
    z = compiled(x)
    assert_equal(y.x, z.x)


def create_new_class_complex(x):
    a = A(x + 1.0)
    a.x = a.x + 1.0
    return a.x + 1.0, a


def test_create_new_class_complex(caplog):
    reset()
    x = torch.randn(4)
    y = create_new_class_complex(x)
    compiled = compile(create_new_class_complex)
    run_and_check_cache(compiled, [MISS, MISS], 1, caplog, x)
    run_and_check_cache(compiled, [HIT], 1, caplog, x)
    z = compiled(x)
    assert_equal(y[0], z[0])
    assert_equal(y[1].x, z[1].x)
