import torch
from common.checker import run_and_check, HIT, MISS, ALL_MISS
from frontend.compile import compile, reset


def view_operation(a):
    shape = (2, 3)
    b = a.view(*shape)
    return b


def reshape_operation(a):
    shape = (2, 3)
    b = a.reshape(*shape)
    return b


def test_call_function_ex(caplog):
    reset()
    compiled2 = compile(view_operation)
    compiled3 = compile(reshape_operation)
    tensor = torch.tensor([1, 2, 3, 4, 5, 6])
    result = view_operation(tensor)
    run_and_check(compiled2, [MISS], 1, caplog, result, tensor)
    run_and_check(compiled2, [HIT], 1, caplog, result, tensor)
    result = reshape_operation(tensor)
    run_and_check(compiled3, [MISS], 2, caplog, result, tensor)
    run_and_check(compiled3, [HIT], 2, caplog, result, tensor)


class closure_call(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.n_heads = 2
        self.key_value_proj_dim = 6

    def forward(self, x):

        def shape(states):
            return states.view(self.n_heads, self.key_value_proj_dim)

        def project(hidden_states):
            hidden_states = shape(hidden_states)
            return hidden_states

        key_states = project(x)
        return key_states


def test_closure_call(caplog):
    reset()
    with torch.no_grad():
        model = closure_call().eval()
        a = torch.arange(12).reshape(3, 4)
        compiled = compile(model)
        expect_result = model(a)
        run_and_check(compiled, [ALL_MISS], 1, caplog, expect_result, a)
        run_and_check(compiled, [HIT], 1, caplog, expect_result, a)
