from frontend.compile import compile, reset
from frontend.utils import add_force_graph_break
from frontend.c_api import get_next_frame_id
from common.checker import run_and_check, HIT, MISS, assert_equal
import torch
import numpy as np


def numpy_to_int(x):
    p = int(np.floor((x - 1) / 2))
    return p


def test_numpy_to_int(caplog):
    reset()
    compiled_numpy_to_int = compile(numpy_to_int)
    result = numpy_to_int(10)
    run_and_check(compiled_numpy_to_int, [MISS], 1, caplog, result, 10)
    run_and_check(compiled_numpy_to_int, [HIT], 1, caplog, result, 10)