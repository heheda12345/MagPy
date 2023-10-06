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

    def forward(self, inp, i, c, h):
        ih = torch.matmul(inp[i], self.weight_ih)
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


class LSTM(nn.Module):

    def __init__(self, batch_size, input_size, hidden_size, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(LSTMCell(input_size, hidden_size))
        for i in range(num_layers):
            self.layers.append(LSTMCell(hidden_size, hidden_size))
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.hidden_size = hidden_size

    def forward(self, inputs):  # seq_len, batch, input_size
        state_c = (
            torch.zeros(self.batch_size, self.hidden_size, device='cuda'),
            torch.zeros(self.batch_size, self.hidden_size, device='cuda'),
            torch.zeros(self.batch_size, self.hidden_size, device='cuda'),
            torch.zeros(self.batch_size, self.hidden_size, device='cuda'),
            torch.zeros(self.batch_size, self.hidden_size, device='cuda'),
            torch.zeros(self.batch_size, self.hidden_size, device='cuda'),
            torch.zeros(self.batch_size, self.hidden_size, device='cuda'),
            torch.zeros(self.batch_size, self.hidden_size, device='cuda'),
            torch.zeros(self.batch_size, self.hidden_size, device='cuda'),
            torch.zeros(self.batch_size, self.hidden_size, device='cuda'),
        )
        state_h = (
            torch.zeros(self.batch_size, self.hidden_size, device='cuda'),
            torch.zeros(self.batch_size, self.hidden_size, device='cuda'),
            torch.zeros(self.batch_size, self.hidden_size, device='cuda'),
            torch.zeros(self.batch_size, self.hidden_size, device='cuda'),
            torch.zeros(self.batch_size, self.hidden_size, device='cuda'),
            torch.zeros(self.batch_size, self.hidden_size, device='cuda'),
            torch.zeros(self.batch_size, self.hidden_size, device='cuda'),
            torch.zeros(self.batch_size, self.hidden_size, device='cuda'),
            torch.zeros(self.batch_size, self.hidden_size, device='cuda'),
            torch.zeros(self.batch_size, self.hidden_size, device='cuda'),
        )
        for i in range(inputs.size()[0]):
            cur_input = inputs[i]
            for j in range(self.num_layers):
                c = state_c[j]
                h = state_h[j]
                ih = torch.matmul(cur_input, self.layers[j].weight_ih)
                hh = torch.matmul(h, self.layers[j].weight_hh)

                ingatei = ih[0]
                forgetgatei = ih[1]
                cellgatei = ih[2]
                outgatei = ih[3]

                ingateh = hh[0]
                forgetgateh = hh[1]
                cellgateh = hh[2]
                outgateh = hh[3]

                ingate1 = ingatei + self.layers[
                    j].bias_ih_0 + ingateh + self.layers[j].bias_hh_0
                ingate = torch.sigmoid(ingate1)

                forgetgate1 = forgetgatei + self.layers[
                    j].bias_ih_1 + forgetgateh + self.layers[j].bias_hh_1
                forgetgate = torch.sigmoid(forgetgate1)

                cellgate1 = cellgatei + self.layers[
                    j].bias_ih_2 + cellgateh + self.layers[j].bias_hh_2
                cellgate = torch.tanh(cellgate1)

                outgate1 = outgatei + self.layers[
                    j].bias_ih_3 + outgateh + self.layers[j].bias_hh_3
                outgate = torch.sigmoid(outgate1)

                c = (forgetgate * c) + (ingate * cellgate)
                h = outgate * torch.tanh(c)

                state_c[j].copy_(c)
                state_h[j].copy_(h)
                cur_input = h
        return state_h[self.num_layers - 1]


def test_lstm_cell(caplog):
    reset()
    with torch.no_grad():
        model = LSTMCell(10, 10).eval()
        inp = torch.randn(10, 10)
        c = torch.randn(10, 10)
        h = torch.randn(10, 10)
        expect_result = model(inp, 1, c, h)
        compiled_model = compile(model)
        run_and_check(compiled_model, [MISS], 1, caplog, expect_result, inp, 1,
                      c, h)
        run_and_check(compiled_model, [HIT], 1, caplog, expect_result, inp, 1,
                      c, h)
        expect_result2 = model(inp, 2, c, h)
        run_and_check(compiled_model, [MISS], 2, caplog, expect_result2, inp, 2,
                      c, h)
        run_and_check(compiled_model, [HIT], 2, caplog, expect_result2, inp, 2,
                      c, h)


def test_lstm_unroll(caplog):
    reset()
    with torch.no_grad():
        seq_len = 4
        num_layers = 10
        hidden_size = 16
        batch_size = 16
        model = LSTM(batch_size, hidden_size, hidden_size, num_layers).cuda()
        model.eval()
        inputs = torch.randn(seq_len, batch_size, hidden_size, device='cuda')
        expect_result = model(inputs)
        compiled = compile(model)
        run_and_check(compiled, [MISS], 1, caplog, expect_result, inputs)
        run_and_check(compiled, [HIT], 1, caplog, expect_result, inputs)
