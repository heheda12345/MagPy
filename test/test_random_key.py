from frontend.utils import new_random_key

import random


def test_random_key():
    random.seed(123)
    a = random.randint(0, 10000)
    b = random.randint(0, 10000)
    random.seed(123)
    aa = random.randint(0, 10000)
    key = new_random_key()
    bb = random.randint(0, 10000)
    assert a == aa
    assert b == bb
