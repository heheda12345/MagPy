from frontend.compile import compile, reset
from frontend.utils import add_force_graph_break
from frontend.c_api import get_next_frame_id
from common.checker import run_and_check, HIT, MISS
import torch


def inplace_add(a, b):
    a += b
    return a


def test_inplace_add(caplog):
    reset()
    compiled = compile(inplace_add)
    result1 = inplace_add(1.0, 2.0)
    run_and_check(compiled, [MISS], 1, caplog, result1, 1.0, 2.0)
    run_and_check(compiled, [HIT], 1, caplog, result1, 1.0, 2.0)

    result2 = inplace_add((1, 2), (3, 4))
    run_and_check(compiled, [MISS], 2, caplog, result2, (1, 2), (3, 4))
    run_and_check(compiled, [HIT], 2, caplog, result2, (1, 2), (3, 4))

    result3 = inplace_add([1, 2], [3, 4])
    run_and_check(compiled, [MISS], 3, caplog, result3, [1, 2], [3, 4])
    run_and_check(compiled, [HIT], 3, caplog, result3, [1, 2], [3, 4])

    result4 = inplace_add(torch.tensor(1), torch.tensor(2))
    run_and_check(compiled, [MISS], 4, caplog, result4, torch.tensor(1),
                  torch.tensor(2))
    result5 = inplace_add(torch.tensor(3), torch.tensor(4))
    run_and_check(compiled, [HIT], 4, caplog, result5, torch.tensor(3),
                  torch.tensor(4))
