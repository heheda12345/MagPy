import pytest
from frontend.compile import compile, reset
import logging
import torch
from common.checker import run_and_check, HIT, MISS


class Model(torch.nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(10, 5)

    def forward(self, x):
        y = self.linear(x)
        z = y + 1.0
        return z


def test_nnmodule(caplog):
    reset()
    with torch.no_grad():
        model = Model().eval()
        x = torch.randn(1, 10)
        expect_result = model(x)
        compiled_model = compile(model)
        run_and_check(compiled_model, [MISS], 2, caplog, expect_result, x)
        run_and_check(compiled_model, [HIT, HIT], 2, caplog, expect_result, x)