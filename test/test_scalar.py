import pytest
from frontend.compile import compile, reset
import logging
from common.checker import run_and_check, HIT, MISS


def perfect0(a):
    return a + 1


def perfect1(a):
    return a + 1


def graph_break(a):
    return (a + 1) // 2


@pytest.fixture
def compiled_perfect0():
    return compile(perfect0)


@pytest.fixture
def compiled_perfect1():
    return compile(perfect1)


@pytest.fixture
def compiled_graph_break():
    return compile(graph_break)


def test_perfect(compiled_perfect0, compiled_perfect1, caplog):
    reset()
    run_and_check(compiled_perfect0, [MISS], 1, caplog, 4, 3)
    run_and_check(compiled_perfect0, [HIT], 1, caplog, 4, 3)
    run_and_check(compiled_perfect0, [MISS], 2, caplog, 5, 4)
    run_and_check(compiled_perfect0, [HIT], 2, caplog, 5, 4)
    run_and_check(compiled_perfect1, [MISS], 3, caplog, 4, 3)
    run_and_check(compiled_perfect1, [HIT], 3, caplog, 4, 3)
    run_and_check(compiled_perfect1, [MISS], 4, caplog, 5, 4)
    run_and_check(compiled_perfect1, [HIT], 4, caplog, 5, 4)


def test_graph_break(compiled_graph_break, caplog):
    reset()
    run_and_check(compiled_graph_break, [MISS], 1, caplog, 2, 3)
    run_and_check(compiled_graph_break, [HIT], 1, caplog, 2, 3)
    run_and_check(compiled_graph_break, [MISS], 2, caplog, 2, 4)
    run_and_check(compiled_graph_break, [HIT], 2, caplog, 2, 4)