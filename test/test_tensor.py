from frontend.compile import compile, reset
from frontend.utils import enable_dyn_shape
from common.checker import run_and_check, HIT, MISS, ALL_MISS
import torch
import torch.utils.checkpoint


def tensor_only(a, b, c):
    return a + b + c


def tensor_add_float(a, b):
    return a + b + 1.0


def tensor_add_int(a, b):
    return a + b + 1


def test_tensor_only(caplog):
    reset()
    compiled_tensor_only = compile(tensor_only)
    a = torch.full((1,), 1.0)
    b = torch.full((1,), 2.0)
    c = torch.full((1,), 3.0)
    result = tensor_only(a, b, c)
    run_and_check(compiled_tensor_only, [MISS], 1, caplog, result, a, b, c)
    run_and_check(compiled_tensor_only, [HIT], 1, caplog, result, a, b, c)
    a = torch.full((2,), 1.0)
    b = torch.full((2,), 2.0)
    c = torch.full((2,), 3.0)
    result = tensor_only(a, b, c)
    run_and_check(compiled_tensor_only, [MISS], 2, caplog, result, a, b, c)
    run_and_check(compiled_tensor_only, [HIT], 2, caplog, result, a, b, c)


def test_with_scalar(caplog):
    reset()
    compiled_tensor_add_float = compile(tensor_add_float)
    compiled_tensor_add_int = compile(tensor_add_int)
    a = torch.full((1,), 1.0)
    b = torch.full((1,), 2.0)
    result = tensor_add_float(a, b)
    run_and_check(compiled_tensor_add_float, [MISS], 1, caplog, result, a, b)
    run_and_check(compiled_tensor_add_float, [HIT], 1, caplog, result, a, b)
    result = tensor_add_int(a, b)
    run_and_check(compiled_tensor_add_int, [MISS], 2, caplog, result, a, b)
    run_and_check(compiled_tensor_add_int, [HIT], 2, caplog, result, a, b)


def tensor_subscr_const(a):
    return a[0] + 1


def tensor_subscr_scalar(a, b):
    return a[b] + 1


def tensor_subscr_none(a):
    return a[None] + 1


def tensor_subscr_tensor(a, b):
    return a[b] + 1


def tensor_subscr_ellipsis(a):
    return a[...] + 1


def tensor_subscr_slice(a):
    s1 = a[:]
    s2 = a[::]
    s3 = a[1:]
    s4 = a[1::]
    s5 = a[:1]
    s6 = a[:1:]
    s7 = a[::2]
    s9 = a[0:1]
    s10 = a[0::2]
    s11 = a[:3:2]
    s12 = a[0:3:2]
    return (s1, s2, s3, s4, s5, s6, s7, s9, s10, s11, s12)


# TODO
def tensor_subscr_tuple(a):
    return a[1, 2] + 1


def test_subscr(caplog):
    reset()
    compiled_tensor_subscr_const = compile(tensor_subscr_const)
    compiled_tensor_subscr_scalar = compile(tensor_subscr_scalar)
    compiled_tensor_subscr_none = compile(tensor_subscr_none)
    compiled_tensor_subscr_tensor = compile(tensor_subscr_tensor)
    compiled_tensor_subscr_slice = compile(tensor_subscr_slice)
    compiled_tensor_subscr_ellipsis = compile(tensor_subscr_ellipsis)
    compiled_tensor_subscr_tuple = compile(tensor_subscr_tuple)
    a = torch.full((3, 3), 1.0)
    b = 0
    idx = torch.tensor([1, 2])

    result = tensor_subscr_const(a)
    run_and_check(compiled_tensor_subscr_const, [MISS], 1, caplog, result, a)
    run_and_check(compiled_tensor_subscr_const, [HIT], 1, caplog, result, a)

    result = tensor_subscr_scalar(a, b)
    run_and_check(compiled_tensor_subscr_scalar, [MISS], 2, caplog, result, a,
                  b)
    run_and_check(compiled_tensor_subscr_scalar, [HIT], 2, caplog, result, a, b)

    result = tensor_subscr_none(a)
    run_and_check(compiled_tensor_subscr_none, [MISS], 3, caplog, result, a)
    run_and_check(compiled_tensor_subscr_none, [HIT], 3, caplog, result, a)

    result = tensor_subscr_tensor(a, idx)
    run_and_check(compiled_tensor_subscr_tensor, [MISS], 4, caplog, result, a,
                  idx)
    run_and_check(compiled_tensor_subscr_tensor, [HIT], 4, caplog, result, a,
                  idx)

    result = tensor_subscr_slice(a)
    run_and_check(compiled_tensor_subscr_slice, [MISS], 5, caplog, result, a)
    run_and_check(compiled_tensor_subscr_slice, [HIT], 5, caplog, result, a)

    # TODO: support ellipsis and tuple after supporting tuple
    result = tensor_subscr_ellipsis(a)
    run_and_check(compiled_tensor_subscr_ellipsis, [MISS], 6, caplog, result, a)
    run_and_check(compiled_tensor_subscr_ellipsis, [HIT], 6, caplog, result, a)

    result = tensor_subscr_tuple(a)
    run_and_check(compiled_tensor_subscr_tuple, [MISS], 7, caplog, result, a)
    run_and_check(compiled_tensor_subscr_tuple, [HIT], 7, caplog, result, a)


def tensor_functional(a):
    return torch.nn.functional.relu(a) + 1


def test_tensor_functional(caplog):
    reset()
    compiled_tensor_functional = compile(tensor_functional)
    a = torch.randn((3, 3))
    expect_result = tensor_functional(a)
    run_and_check(compiled_tensor_functional, [MISS], 1, caplog, expect_result,
                  a)
    run_and_check(compiled_tensor_functional, [HIT], 1, caplog, expect_result,
                  a)


def fx_nest(x):
    return torch.cat([x] + [x.mul(0)] * 2, 1)


def test_fx_nest(caplog):
    reset()
    compiled_fx_nest = compile(fx_nest)
    a = torch.randn((3, 3))
    expect_result = fx_nest(a)
    run_and_check(compiled_fx_nest, [MISS], 1, caplog, expect_result, a)
    run_and_check(compiled_fx_nest, [HIT], 1, caplog, expect_result, a)
    b = torch.randn((3, 3))
    expect_result = fx_nest(b)
    run_and_check(compiled_fx_nest, [HIT], 1, caplog, expect_result, b)


def tensor_shape(a):
    return a.size(), a.shape


def test_tensor_shape(caplog):
    reset()
    compiled_tensor_shape = compile(tensor_shape)
    a = torch.randn((3, 3))
    expect_result = tensor_shape(a)
    run_and_check(compiled_tensor_shape, [MISS], 1, caplog, expect_result, a)
    run_and_check(compiled_tensor_shape, [HIT], 1, caplog, expect_result, a)


def tensor_dtype(a):
    return a.dtype


def test_tensor_dtype(caplog):
    reset()
    compiled_tensor_dtype = compile(tensor_dtype)
    a = torch.randn((3, 3))
    expect_result = tensor_dtype(a)
    run_and_check(compiled_tensor_dtype, [MISS], 1, caplog, expect_result, a)
    run_and_check(compiled_tensor_dtype, [HIT], 1, caplog, expect_result, a)


def tensor_type(a):
    return a.data.type(a.dtype)


def test_tensor_type(caplog):
    reset()
    compiled_tensor_type = compile(tensor_type)
    a = torch.randn((4, 4))
    result = tensor_type(a)
    run_and_check(compiled_tensor_type, [MISS], 1, caplog, result, a)
    run_and_check(compiled_tensor_type, [HIT], 1, caplog, result, a)


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
            run_and_check(compiled, [ALL_MISS], 1, caplog, y1, inp1)
            run_and_check(compiled, [HIT], 1, caplog, y1, inp1)
            run_and_check(compiled, [HIT], 1, caplog, y2, inp2)


def run_tensor_new(x):
    mask = x.data.new().resize_as_(x.data).fill_(0)
    return mask


def test_tensor_new(caplog):
    reset()
    with torch.no_grad():
        inp = torch.rand((3, 3))
        expect = run_tensor_new(inp)
        compiled = compile(run_tensor_new)
        run_and_check(compiled, [MISS], 1, caplog, expect, inp)
        run_and_check(compiled, [HIT], 1, caplog, expect, inp)


def iter_f1(x):
    s = x[0]
    for y in x:
        s = s + y
    return s


def iter_f2(x):
    s = x[0]
    for i, y in enumerate(x):
        s = s + y + i
    return s


def test_tensor_iter(caplog):
    reset()
    with torch.no_grad():
        inp = torch.rand((3, 3))
        expect = iter_f1(inp)
        compiled = compile(iter_f1)
        run_and_check(compiled, [MISS, MISS], 1, caplog, expect, inp)
        run_and_check(compiled, [HIT], 1, caplog, expect, inp)

        expect = iter_f2(inp)
        compiled = compile(iter_f2)
        run_and_check(compiled, [MISS, MISS], 2, caplog, expect, inp)
        run_and_check(compiled, [HIT], 2, caplog, expect, inp)


def tensor_item(x):
    return x.item() + 1


def test_tensor_item(caplog):
    reset()
    inp = torch.rand((1,))
    expect = tensor_item(inp)
    compiled = compile(tensor_item)
    run_and_check(compiled, [MISS], 1, caplog, expect, inp)
    run_and_check(compiled, [HIT], 1, caplog, expect, inp)


def ex_callee(x, y):
    return x + y


def ex_caller(x):
    return ex_callee(*x)


def test_tensor_call_ex(caplog):
    reset()
    with torch.no_grad():
        inp = torch.rand((2, 3))
        expect = ex_caller(inp)
        compiled = compile(ex_caller)
        run_and_check(compiled, [ALL_MISS], 1, caplog, expect, inp)
        run_and_check(compiled, [HIT], 1, caplog, expect, inp)


def run_get_device_states(x):
    return torch.utils.checkpoint.get_device_states(*x)


def test_get_device_states(caplog):
    reset()
    from frontend.utils import SetConfig
    with torch.no_grad():
        with SetConfig({"backend": "eager"}):
            inp = torch.rand((2, 3)).cuda()
            expect = run_get_device_states(inp)
            compiled = compile(run_get_device_states)
            run_and_check(compiled, [ALL_MISS], 1, caplog, expect, inp)
            run_and_check(compiled, [HIT], 1, caplog, expect, inp)


# not yet support due to yield
# def tuple_view1(a):
#     # out = tuple(i.view(4, 3) for i in a)
#     out = tuple(i.view(4, 3) for i in a)
#     return out[0]

# def tuple_view2_callee(a):
#     out = tuple(i.view(4, 3) for i in a)
#     return out

# def tuple_view2_caller(a):
#     x, y = tuple_view2_callee(a)
#     return x + y

# def test_tuple_view(caplog):
#     reset()
#     a = (torch.randn([3, 4]),)
#     expect = tuple_view1(a)
#     compiled = compile(tuple_view1)
#     run_and_check(compiled, [ALL_MISS], 1, caplog, expect, a)
#     run_and_check(compiled, [HIT], 1, caplog, expect, a)


def run_getattr_relu(x):
    func = getattr(torch.nn.functional, 'relu')
    return func(x)


def test_run_getattr_relu(caplog):
    reset()
    with torch.no_grad():
        inp = torch.rand((2, 2))
        expect = run_getattr_relu(inp)
        compiled = compile(run_getattr_relu)
        run_and_check(compiled, [ALL_MISS], 1, caplog, expect, inp)
        run_and_check(compiled, [HIT], 1, caplog, expect, inp)


def run_type_tensor(x):
    return x.type(torch.LongTensor)


def test_run_type_tensor(caplog):
    reset()
    with torch.no_grad():
        inp = torch.rand((2, 2))
        expect = run_type_tensor(inp)
        compiled = compile(run_type_tensor)
        run_and_check(compiled, [MISS], 1, caplog, expect, inp)
        run_and_check(compiled, [HIT], 1, caplog, expect, inp)


# def run_no_grad(x):
#     with torch.no_grad():
#         y = x * 2
#     return y

# def test_no_grad(caplog):
#     reset()
#     with torch.no_grad():
#         inp = torch.rand((2, 2))
#         expect = run_no_grad(inp)
#         compiled = compile(run_no_grad)
#         run_and_check(compiled, [MISS], 1, caplog, expect, inp)
#         run_and_check(compiled, [HIT], 1, caplog, expect, inp)
