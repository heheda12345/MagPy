import pytest
from frontend.compile import compile, reset
import logging
from common.checker import run_and_check, HIT, MISS


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
    compiled_graph_break1 = compile(graph_break1)
    run_and_check(compiled_graph_break0, [MISS], 1, caplog, 2, 3)
    run_and_check(compiled_graph_break0, [HIT], 1, caplog, 2, 3)
    run_and_check(compiled_graph_break0, [MISS], 2, caplog, 2, 4)
    run_and_check(compiled_graph_break0, [HIT], 2, caplog, 2, 4)
    run_and_check(compiled_graph_break1, [MISS], 4, caplog, 2, 1)
    run_and_check(compiled_graph_break1, [HIT, HIT], 4, caplog, 2, 1)
    run_and_check(compiled_graph_break1, [MISS, HIT], 5, caplog, 2, 2)
    run_and_check(compiled_graph_break1, [HIT, HIT], 5, caplog, 2, 2)
