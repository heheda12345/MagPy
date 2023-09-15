def fn(a, b, c, d):
    return a + b, c + d


def test_int_cache():
    a, b = fn(1, 4, 2, 3)
    assert id(a) != id(b)