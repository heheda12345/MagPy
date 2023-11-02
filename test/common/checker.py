from enum import Enum
import logging
from collections import Iterable
import torch
from frontend import cache

HIT = 1
MISS = 2
ALL_MISS = 3


def assert_equal(ref, out):
    precision = 1e-3
    assert type(ref) == type(
        out), f"wrong type: expect {type(ref)}, got {type(out)}"
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
        if isinstance(ref, set):
            assert (len(ref) == len(out))
        else:
            for r, o in zip(ref, out):
                assert_equal(r, o)
    else:
        assert ref == out, f"wrong answer: expect {ref}, got {out}"


def check_cache_log(caplog, expect_cache_logs, expect_cache_size: int):
    recorded_cache_logs = []
    for record in caplog.records:
        if record.message.startswith("\033[31mguard cache"):
            if "hit" in record.message:
                recorded_cache_logs.append(HIT)
            elif "miss" in record.message:
                recorded_cache_logs.append(MISS)
            else:
                assert (False), "unknown cache log"
    if len(expect_cache_logs) == 1 and expect_cache_logs[0] == ALL_MISS:
        expect_cache_logs = [MISS for _ in range(len(recorded_cache_logs))]
    assert len(recorded_cache_logs) == len(
        expect_cache_logs
    ), f"wrong cache log: expect {expect_cache_logs}, got {recorded_cache_logs}"
    for recorded, expected in zip(recorded_cache_logs, expect_cache_logs):
        assert recorded == expected, f"wrong cache log: expect {expect_cache_logs}, got {recorded_cache_logs}"
    assert cache.TOTAL_SIZE == expect_cache_size, f"wrong cache size: expect {expect_cache_size}, got {cache.TOTAL_SIZE}"


def should_not_call(*args, **kwargs):
    raise ValueError("should not rewrite bytecode")


class DisableRewriteByteCode:
    old_should_call: bool

    def __enter__(self):
        from frontend import bytecode_writter
        self.old_should_call = bytecode_writter.SHOULD_NOT_CALL_REWRITE
        bytecode_writter.SHOULD_NOT_CALL_REWRITE = True

    def __exit__(self, exc_type, exc_value, traceback):
        from frontend import bytecode_writter
        bytecode_writter.SHOULD_NOT_CALL_REWRITE = self.old_should_call


def run_and_check(compiled, expect_cache_logs, expect_cache_size: int, caplog,
                  expected_result, *args, **kwargs):
    caplog.set_level(logging.INFO)
    caplog.clear()
    with torch.no_grad():
        if all([x == HIT for x in expect_cache_logs]):
            with DisableRewriteByteCode():
                out = compiled(*args, **kwargs)
        else:
            out = compiled(*args, **kwargs)
    assert_equal(expected_result, out)
    check_cache_log(caplog, expect_cache_logs, expect_cache_size)


def run_and_check_cache(compiled, expect_cache_logs, expect_cache_size: int,
                        caplog, *args, **kwargs):  # do not perform result check
    caplog.set_level(logging.INFO)
    caplog.clear()
    with torch.no_grad():
        if all([x == HIT for x in expect_cache_logs]):
            with DisableRewriteByteCode():
                _ = compiled(*args, **kwargs)
        else:
            _ = compiled(*args, **kwargs)
    check_cache_log(caplog, expect_cache_logs, expect_cache_size)
