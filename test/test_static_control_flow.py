import pytest
from frontend.compile import compile, reset
from frontend.utils import add_force_graph_break
from frontend.c_api import get_next_frame_id
import logging
from common.checker import run_and_check, HIT, MISS

import torch
import torch.nn as nn


class Model(torch.nn.Module):

    def __init__(self, bn):
        super().__init__()
        self.a = torch.nn.Parameter(torch.randn(2, 2))
        if bn:
            self.bn = torch.nn.BatchNorm1d(2)
        else:
            self.bn = None

    def forward(self, x):
        y = self.a * x
        if self.bn:
            y = self.bn(y)
        return y


def test_static_cf(caplog):
    reset()
    x = torch.randn(2, 2)
    model1 = Model(True).eval()
    expect_result = model1(x)
    compiled_model1 = compile(model1)
    run_and_check(compiled_model1, [MISS], 1, caplog, expect_result, x)
    run_and_check(compiled_model1, [HIT], 1, caplog, expect_result, x)

    model2 = Model(False).eval()
    expect_result = model2(x)
    compiled_model2 = compile(model2)
    run_and_check(compiled_model2, [MISS], 2, caplog, expect_result, x)
    run_and_check(compiled_model2, [HIT], 2, caplog, expect_result, x)

    reset()
    model1 = Model(True).eval()
    expect_result1 = model1(x)
    add_force_graph_break(get_next_frame_id() - 1, 8)  # frame of model1
    compiled_model1 = compile(model1)
    run_and_check(compiled_model1, [MISS], 2, caplog, expect_result1, x)
    run_and_check(compiled_model1, [HIT, HIT], 2, caplog, expect_result1, x)

    model2 = Model(False).eval()
    expect_result2 = model2(x)
    compiled_model2 = compile(model2)
    run_and_check(compiled_model2, [MISS], 3, caplog, expect_result2, x)
    run_and_check(compiled_model2, [HIT], 3, caplog, expect_result2, x)

    run_and_check(compiled_model1, [HIT, HIT], 3, caplog, expect_result1, x)