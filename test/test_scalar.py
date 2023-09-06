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
    compiled_graph_break0 = compile(graph_break0_float)
    compiled_graph_break1 = compile(graph_break1_float)
    run_and_check(compiled_graph_break0, [MISS], 1, caplog, 2.0, 3.0)
    run_and_check(compiled_graph_break0, [HIT], 1, caplog, 2.0, 3.0)
    run_and_check(compiled_graph_break0, [MISS], 2, caplog, 2.5, 4.0)
    run_and_check(compiled_graph_break0, [HIT], 2, caplog, 2.5, 4.0)
    run_and_check(compiled_graph_break1, [MISS], 4, caplog, 2.0, 1.0)
    run_and_check(compiled_graph_break1, [HIT, HIT], 4, caplog, 2.0, 1.0)
