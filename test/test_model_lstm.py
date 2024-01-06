import pytest
from frontend.compile import compile, reset
from frontend.utils import add_force_graph_break
from frontend.c_api import get_next_frame_id
import logging
from common.checker import run_and_check, HIT, MISS, ALL_MISS
from frontend.dynamic import DynamicControlFlow, mark_dynamic_pc
from frontend.utils import SetConfig
import torch
import torch.nn as nn
from torch.nn import Parameter
from torch import Tensor
from typing import Tuple


class MyLSTMCell(nn.Module):

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


class MyLSTM(nn.Module):

    def __init__(self, batch_size, input_size, hidden_size, num_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(MyLSTMCell(input_size, hidden_size))
        for i in range(num_layers):
            self.layers.append(MyLSTMCell(hidden_size, hidden_size))
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


@pytest.mark.model
def test_my_lstm_cell(caplog):
    reset()
    with torch.no_grad():
        model = MyLSTMCell(10, 10).eval()
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


@pytest.mark.model
def test_my_lstm_loop(caplog):
    reset()
    with SetConfig({"backend": "eager"}):
        with torch.no_grad():
            seq_len = 2
            num_layers = 2
            hidden_size = 256
            batch_size = 16
            model = MyLSTM(batch_size, hidden_size, hidden_size,
                           num_layers).cuda()
            model.eval()
            inputs = torch.randn(seq_len,
                                 batch_size,
                                 hidden_size,
                                 device='cuda')
            expect_result = model(inputs)
            for_iter_pc = 193
            mark_dynamic_pc(get_next_frame_id(), for_iter_pc,
                            DynamicControlFlow(for_iter_pc, "FOR_ITER"))
            compiled = compile(model)
            run_and_check(compiled, [MISS], 1, caplog, expect_result, inputs)
            run_and_check(compiled, [HIT], 1, caplog, expect_result, inputs)


@pytest.mark.model
def test_my_lstm_unroll(caplog):
    reset()
    with torch.no_grad():
        seq_len = 4
        num_layers = 10
        hidden_size = 16
        batch_size = 16
        model = MyLSTM(batch_size, hidden_size, hidden_size, num_layers).cuda()
        model.eval()
        inputs = torch.randn(seq_len, batch_size, hidden_size, device='cuda')
        expect_result = model(inputs)
        compiled = compile(model)
        run_and_check(compiled, [MISS], 1, caplog, expect_result, inputs)
        run_and_check(compiled, [HIT], 1, caplog, expect_result, inputs)


class SingleLayerRNN(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(SingleLayerRNN, self).__init__()
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(
            torch.randn(input_size, hidden_size, dtype=torch.float32))
        self.weight_hh = nn.Parameter(
            torch.randn(hidden_size, hidden_size, dtype=torch.float32))

    def forward(self, x):
        h = torch.zeros(x.shape[1], self.hidden_size)
        for i in range(x.size()[0]):
            h = torch.matmul(x[i], self.weight_ih) + torch.matmul(
                h, self.weight_hh)
            h = torch.tanh(h)
        return h


def test_rnn_break(caplog):
    reset()
    with torch.no_grad():
        hidden_size = 4
        seq_len = 3
        batch_size = 2
        model = SingleLayerRNN(hidden_size, hidden_size).eval()
        inputs = torch.randn(seq_len, batch_size, hidden_size)
        expected = model(inputs)
        print(expected)
        mark_dynamic_pc(0, 18, DynamicControlFlow(18, "FOR_ITER"))
        add_force_graph_break(get_next_frame_id(), 27)
        compiled = compile(model)
        run_and_check(compiled, [MISS], 4, caplog, expected, inputs)
        run_and_check(compiled, [HIT, HIT, HIT, HIT], 4, caplog, expected,
                      inputs)


def test_rnn_no_break(caplog):
    reset()
    with torch.no_grad():
        hidden_size = 4
        seq_len = 3
        batch_size = 2
        model = SingleLayerRNN(hidden_size, hidden_size).eval()
        inputs = torch.randn(seq_len, batch_size, hidden_size)
        expected = model(inputs)
        print(expected)
        mark_dynamic_pc(0, 18, DynamicControlFlow(18, "FOR_ITER"))
        compiled = compile(model)
        run_and_check(compiled, [MISS], 1, caplog, expected, inputs)
        run_and_check(compiled, [HIT], 1, caplog, expected, inputs)


class LSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.bias_ih = Parameter(torch.randn(4 * hidden_size))
        self.bias_hh = Parameter(torch.randn(4 * hidden_size))

    def forward(
            self, input: Tensor,
            state: Tuple[Tensor,
                         Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        hx, cx = state
        gates = (torch.mm(input, self.weight_ih.t()) + self.bias_ih +
                 torch.mm(hx, self.weight_hh.t()) + self.bias_hh)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        return hy, (hy, cy)


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
        state_c = [
            torch.zeros(self.batch_size, self.hidden_size, device='cuda')
            for _ in range(self.num_layers)
        ]
        state_h = [
            torch.zeros(self.batch_size, self.hidden_size, device='cuda')
            for _ in range(self.num_layers)
        ]
        for i in range(inputs.size()[0]):
            cur_input = inputs[i]
            for j in range(self.num_layers):
                c = state_c[j]
                h = state_h[j]
                _, (h, c) = self.layers[j](cur_input, (h, c))
                state_c[j].copy_(c)
                state_h[j].copy_(h)
                cur_input = h
        return state_h[self.num_layers - 1]


@pytest.mark.model
def test_lstm_loop(caplog):
    reset()
    with SetConfig({"backend": "eager"}):
        with torch.no_grad():
            seq_len = 2
            num_layers = 2
            hidden_size = 256
            batch_size = 16
            model = LSTM(batch_size, hidden_size, hidden_size,
                         num_layers).cuda()
            model.eval()
            inputs = torch.randn(seq_len,
                                 batch_size,
                                 hidden_size,
                                 device='cuda')
            expect_result = model(inputs)
            for_iter_pc = 193
            mark_dynamic_pc(get_next_frame_id(), for_iter_pc,
                            DynamicControlFlow(for_iter_pc, "FOR_ITER"))
            compiled = compile(model)
            run_and_check(compiled, [ALL_MISS], 1, caplog, expect_result,
                          inputs)
            run_and_check(compiled, [HIT], 1, caplog, expect_result, inputs)


@pytest.mark.model
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
        run_and_check(compiled, [ALL_MISS], 1, caplog, expect_result, inputs)
        run_and_check(compiled, [HIT], 1, caplog, expect_result, inputs)


# generated/test_BangLiu_QANet_PyTorch.py
class RNN(nn.Module):
    """
    General Recurrent Neural Network module.
    Input: tensor of shape (seq_len, batch, input_size)
    Output: tensor of shape (seq_len, batch, hidden_size * num_directions)
    """

    def __init__(self,
                 input_size,
                 hidden_size,
                 output_projection_size=None,
                 num_layers=1,
                 bidirectional=True,
                 cell_type='lstm',
                 dropout=0,
                 pack=False,
                 batch_first=False,
                 init_method='default'):
        super().__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        if output_projection_size is not None:
            self.output_layer = nn.Linear(
                hidden_size * 2 if bidirectional else hidden_size,
                output_projection_size)
        self.pack = pack
        network = self._get_rnn(cell_type)
        self.network = network(input_size=input_size,
                               hidden_size=hidden_size,
                               num_layers=num_layers,
                               bidirectional=bidirectional,
                               dropout=dropout,
                               batch_first=batch_first)

    def forward(self, input_variable):
        outputs, hidden = self.network(input_variable)
        if self.pack:
            padded_outputs, lengths = pad_packed_sequence(outputs)
            if hasattr(self, 'output_layer'):
                outputs = pack_padded_sequence(
                    self.output_layer(padded_outputs), lengths)
        elif hasattr(self, 'output_layer'):
            outputs = self.output_layer(outputs)
        return outputs, hidden

    def _get_rnn(self, rnn_type):
        rnn_type = rnn_type.lower()
        if rnn_type == 'gru':
            network = torch.nn.GRU
        elif rnn_type == 'lstm':
            network = torch.nn.LSTM
        else:
            raise ValueError('Invalid RNN type %s' % rnn_type)
        return network


def test_builtin_rnn(caplog):
    reset()
    with torch.no_grad():
        model = RNN(4, 4)
        x = torch.randn(4, 4)
        expect_result = model(x)
        compiled_model = compile(model)
        run_and_check(compiled_model, [MISS], 1, caplog, expect_result, x)
        run_and_check(compiled_model, [HIT], 1, caplog, expect_result, x)
