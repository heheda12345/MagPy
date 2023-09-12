import pytest
from frontend.compile import compile, reset
from frontend.utils import add_force_graph_break
from frontend.c_api import get_next_frame_id
import logging
from common.checker import run_and_check, HIT, MISS

import torch
import torch.nn as nn


class LSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.weight_ih = nn.Parameter(
            torch.randn(4, input_size, hidden_size, dtype=torch.float32))
        self.weight_hh = nn.Parameter(
            torch.randn(4, hidden_size, hidden_size, dtype=torch.float32))
        self.bias_ih_0 = nn.Parameter(
            torch.randn(hidden_size, dtype=torch.float32))
        self.bias_hh_0 = nn.Parameter(
            torch.randn(hidden_size, dtype=torch.float32))
        self.bias_ih_1 = nn.Parameter(
            torch.randn(hidden_size, dtype=torch.float32))
        self.bias_hh_1 = nn.Parameter(
            torch.randn(hidden_size, dtype=torch.float32))
        self.bias_ih_2 = nn.Parameter(
            torch.randn(hidden_size, dtype=torch.float32))
        self.bias_hh_2 = nn.Parameter(
            torch.randn(hidden_size, dtype=torch.float32))
        self.bias_ih_3 = nn.Parameter(
            torch.randn(hidden_size, dtype=torch.float32))
        self.bias_hh_3 = nn.Parameter(
            torch.randn(hidden_size, dtype=torch.float32))
        self.hidden_size = hidden_size
        nn.init.xavier_uniform_(self.weight_ih)
        nn.init.xavier_uniform_(self.weight_hh)

    def forward(self, inp, c, h):
        ih = torch.matmul(inp, self.weight_ih)
        hh = torch.matmul(h, self.weight_hh)

        ingatei = ih[0]
        forgetgatei = ih[1]
        cellgatei = ih[2]
        outgatei = ih[3]

        ingateh = hh[0]
        forgetgateh = hh[1]
        cellgateh = hh[2]
        outgateh = hh[3]

        ingate1 = ingatei + self.bias_ih_0 + ingateh + self.bias_hh_0
        ingate = torch.sigmoid(ingate1)

        forgetgate1 = forgetgatei + self.bias_ih_1 + forgetgateh + self.bias_hh_1
        forgetgate = torch.sigmoid(forgetgate1)

        cellgate1 = cellgatei + self.bias_ih_2 + cellgateh + self.bias_hh_2
        cellgate = torch.tanh(cellgate1)

        outgate1 = outgatei + self.bias_ih_3 + outgateh + self.bias_hh_3
        outgate = torch.sigmoid(outgate1)

        c = (forgetgate * c) + (ingate * cellgate)
        h = outgate * torch.tanh(c)
        return c, h


def test_lstm_cell(caplog):
    reset()
    with torch.no_grad():
        model = LSTMCell(10, 10).eval()
        inp = torch.randn(10, 10)
        c = torch.randn(10, 10)
        h = torch.randn(10, 10)
        expect_result = model(inp, c, h)
        compiled_model = compile(model)

        run_and_check(compiled_model, [MISS], 1, caplog, expect_result, inp, c,
                      h)
        run_and_check(compiled_model, [HIT], 1, caplog, expect_result, inp, c,
                      h)
