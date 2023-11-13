from frontend.compile import compile, reset
from common.checker import run_and_check, HIT, MISS
import torch


def tensor_only(a, b, c):
    return a + b + c


def tensor_add_float(a, b):
    return a + b + 1.0


def tensor_add_int(a, b):
    return a + b + 1


def test_tensor_only(caplog):
    reset()
    compiled_tensor_only = compile(tensor_only)
    a = torch.full((1, ), 1.0)
    b = torch.full((1, ), 2.0)
    c = torch.full((1, ), 3.0)
    result = tensor_only(a, b, c)
    run_and_check(compiled_tensor_only, [MISS], 1, caplog, result, a, b, c)
    run_and_check(compiled_tensor_only, [HIT], 1, caplog, result, a, b, c)
    a = torch.full((2, ), 1.0)
    b = torch.full((2, ), 2.0)
    c = torch.full((2, ), 3.0)
    result = tensor_only(a, b, c)
    run_and_check(compiled_tensor_only, [MISS], 2, caplog, result, a, b, c)
    run_and_check(compiled_tensor_only, [HIT], 2, caplog, result, a, b, c)


def test_with_scalar(caplog):
    reset()
    compiled_tensor_add_float = compile(tensor_add_float)
    compiled_tensor_add_int = compile(tensor_add_int)
    a = torch.full((1, ), 1.0)
    b = torch.full((1, ), 2.0)
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
    run_and_check(compiled_tensor_subscr_scalar, [HIT], 2, caplog, result, a,
                  b)

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
