import torch
from frontend.compile import compile, reset
from common.checker import run_and_check, HIT, MISS, assert_equal


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