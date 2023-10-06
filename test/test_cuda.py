from frontend.compile import compile, reset
from frontend.utils import add_force_graph_break
from frontend.c_api import get_next_frame_id
from common.checker import run_and_check, HIT, MISS
import torch


def simple_add(a):
    return a + 1


def test_simple_add(caplog):
    reset()
    a = torch.full((1, 1), 1.0).cuda()
    expected = simple_add(a)
    compiled = compile(simple_add)
    run_and_check(compiled, [MISS], 1, caplog, expected, a)
    run_and_check(compiled, [HIT], 1, caplog, expected, a)
