from enum import Enum
import logging
from collections import Iterable
import torch
from frontend import cache

HIT = 1
MISS = 2


def assert_equal(ref, out):
    precision = 1e-3
    if isinstance(ref, torch.Tensor):
        assert (isinstance(out, torch.Tensor))
        r = ref.cpu()
        o = out.cpu()
        if r.dtype == torch.bool and o.dtype == torch.int8:
            o = o.bool()
        all_close = torch.allclose(r, o, atol=precision, rtol=precision)
        if not all_close:
            close = torch.isclose(r, o, rtol=precision, atol=precision)
            print("ref:", torch.masked_select(r, ~close))
            print("out:", torch.masked_select(o, ~close))
            print(torch.sum(~close))
            print("wrong answer !!!!!!!!!!!!!!!!!!!!!!!!!!")
            assert (False)
    elif isinstance(ref, Iterable):
        assert (isinstance(out, Iterable))
        for r, o in zip(ref, out):
            assert_equal(r, o)
    else:
        assert ref == out, f"wrong answer: expect {ref}, got {out}"


def run_and_check(compiled, expect_cache_logs, expect_cache_size: int, caplog,
                  expected_result, *args, **kwargs):
    caplog.set_level(logging.INFO)
    caplog.clear()

    out = compiled(*args, **kwargs)
    assert_equal(expected_result, out)
    recorded_cache_logs = []
    for record in caplog.records:
        if record.message.startswith("guard cache"):
            if "hit" in record.message:
                recorded_cache_logs.append(HIT)
            elif "miss" in record.message:
                recorded_cache_logs.append(MISS)
            else:
                assert (False), "unknown cache log"
    assert len(recorded_cache_logs) == len(expect_cache_logs)
    for recorded, expected in zip(recorded_cache_logs, expect_cache_logs):
        assert recorded == expected, f"wrong cache log: expect {expect_cache_logs}, got {recorded_cache_logs}"
    assert cache.TOTAL_SIZE == expect_cache_size, f"wrong cache size: expect {expect_cache_size}, got {cache.TOTAL_SIZE}"