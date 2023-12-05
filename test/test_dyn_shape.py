from frontend.compile import compile, reset
from frontend.utils import enable_dyn_shape
from common.checker import run_and_check, HIT, MISS, ALL_MISS, assert_equal
import torch


def dyn_shape1(a):
    b = a * 2
    return b.view((b.shape[0] * 2, 2))


def dyn_shape2(a):
    return a.view((a.shape[0] * 2, 2)) * 2


def dyn_callee(sz):
    return torch.ones((sz,))


def dyn_caller(a):
    b = a * 2
    return dyn_callee(b.size(0))


def test_dyn_shape(caplog):
    reset()
    for i, fn in enumerate((dyn_shape1, dyn_shape2, dyn_caller)):
        with enable_dyn_shape():
            inp1 = torch.randn((5, 2, 2))
            y1 = fn(inp1)
            inp2 = torch.randn((10, 2, 2))
            y2 = fn(inp2)

            compiled = compile(fn)
            run_and_check(compiled, [ALL_MISS], i + 1, caplog, y1, inp1)
            run_and_check(compiled, [HIT], i + 1, caplog, y1, inp1)
            run_and_check(compiled, [HIT], i + 1, caplog, y2, inp2)


class Model(torch.nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.linear = torch.nn.Linear(3, 3, bias=False)

    def forward(self, x):
        return self.linear(x * 2)


def test_dyn_module(caplog):
    reset()
    with enable_dyn_shape():
        with torch.no_grad():
            model = Model().cuda().eval()
            inp1 = torch.randn((4, 5, 3)).cuda()
            y1 = model(inp1)
            inp2 = torch.randn((4, 5, 3)).cuda()
            y2 = model(inp2)

            compiled = compile(model)
            run_and_check(compiled, [MISS], 1, caplog, y1, inp1)
            run_and_check(compiled, [HIT], 1, caplog, y1, inp1)
            run_and_check(compiled, [HIT], 1, caplog, y2, inp2)


class ModelParam(torch.nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.param = torch.nn.Parameter(torch.randn((3, 3)))

    def forward(self, x):
        return x * self.param


def test_dyn_module_param(caplog):
    reset()
    with enable_dyn_shape():
        with torch.no_grad():
            model = ModelParam().eval()
            inp1 = torch.randn((4, 3, 3))
            y1 = model(inp1)
            inp2 = torch.randn((4, 3, 3))
            y2 = model(inp2)

            compiled = compile(model)
            run_and_check(compiled, [MISS], 1, caplog, y1, inp1)
            run_and_check(compiled, [HIT], 1, caplog, y1, inp1)
            run_and_check(compiled, [HIT], 1, caplog, y2, inp2)


def shape_min_max(a, b, sz):
    x = min(a.size(0), b.size(0))
    y = max(a.size(1), b.size(1))
    z = max(a.size(2), sz)
    return torch.ones((x, y, z))


def test_shape_min_max(caplog):
    reset()
    with enable_dyn_shape():
        with torch.no_grad():
            x1 = torch.randn((4, 3, 5)).cuda()
            y1 = torch.randn((4, 3, 5)).cuda()
            z1 = 7
            out1 = shape_min_max(x1, y1, z1)
            x2 = torch.randn((5, 3, 5)).cuda()
            y2 = torch.randn((4, 3, 5)).cuda()
            z2 = 5
            out2 = shape_min_max(x2, y2, z2)

            compiled = compile(shape_min_max)
            run_and_check(compiled, [MISS], 1, caplog, out1, x1, y1, z1)
            run_and_check(compiled, [HIT], 1, caplog, out1, x1, y1, z1)
            run_and_check(compiled, [MISS], 2, caplog, out2, x2, y2, z2)
            run_and_check(compiled, [HIT], 2, caplog, out2, x2, y2, z2)


def dyn_slice_callee(a, sz):
    return a[:sz]


def dyn_slice_caller(a):
    b = a * 2
    return dyn_slice_callee(b, b.size(0))


def test_dyn_slice(caplog):
    reset()
    with enable_dyn_shape():
        with torch.no_grad():
            x1 = torch.randn((4, 3, 5)).cuda()
            out1 = dyn_slice_caller(x1)
            x2 = torch.randn((8, 3, 5)).cuda()
            out2 = dyn_slice_caller(x2)

            compiled = compile(dyn_slice_caller)
            run_and_check(compiled, [MISS, MISS], 1, caplog, out1, x1)
            run_and_check(compiled, [HIT], 1, caplog, out1, x1)
            run_and_check(compiled, [HIT], 1, caplog, out2, x2)
