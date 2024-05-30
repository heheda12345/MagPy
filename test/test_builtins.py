import pytest
from frontend.compile import compile, reset
from frontend.utils import add_force_graph_break
from frontend.c_api import get_next_frame_id
import logging
from common.checker import run_and_check, HIT, MISS


def run_enumerate(x):
    s = 0
    for i, v in enumerate(x):
        s += i * v
    return s, enumerate(x)


def run_enumerate2(x):
    s = 0
    for i, v in enumerate(x, 2):
        s += i * v
    return s


def test_enumerate(caplog):
    reset()
    compiled_run_enumerate = compile(run_enumerate)
    expect_result = run_enumerate([1, 2, 3, 4, 5])
    run_and_check(compiled_run_enumerate, [MISS], 1, caplog, expect_result,
                  [1, 2, 3, 4, 5])
    expect_result = run_enumerate([1, 2, 3, 4, 5])
    run_and_check(compiled_run_enumerate, [HIT], 1, caplog, expect_result,
                  [1, 2, 3, 4, 5])
    compiled_run_enumerate2 = compile(run_enumerate2)
    expect_result2 = run_enumerate2([1, 2, 3, 4, 5])
    run_and_check(compiled_run_enumerate2, [MISS], 2, caplog, expect_result2,
                  [1, 2, 3, 4, 5])
    run_and_check(compiled_run_enumerate2, [HIT], 2, caplog, expect_result2,
                  [1, 2, 3, 4, 5])
