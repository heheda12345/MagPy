def test_int_cache():
    a = 1
    b = 1
    assert id(a) != id(b)