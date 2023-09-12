from frontend.compile import compile, reset
from frontend.utils import add_force_graph_break
from frontend.c_api import get_next_frame_id
from common.checker import run_and_check, HIT, MISS
import torch


def store_no_break(a, b):
    c = a + b
    return c


def store_with_break(a, b):
    c = a + b
    d = c + a
    e = c + d
    f = c / 2 + e
    return f


def test_store(caplog):
    reset()
    compiled_store_no_break = compile(store_no_break)
    a = torch.full((1,), 1.0)
    b = torch.full((1,), 2.0)
    result = store_no_break(a, b)
    run_and_check(compiled_store_no_break, [MISS], 1, caplog, result, a, b)
    run_and_check(compiled_store_no_break, [HIT], 1, caplog, result, a, b)

    compiled_store_with_break = compile(store_with_break)
    add_force_graph_break(get_next_frame_id(), 14)
    result = store_with_break(a, b)
    run_and_check(compiled_store_with_break, [MISS], 3, caplog, result, a, b)
    run_and_check(compiled_store_with_break, [HIT, HIT], 3, caplog, result, a,
                  b)
