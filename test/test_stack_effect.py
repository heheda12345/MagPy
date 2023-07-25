import dis
from frontend.c_api import stack_effect


def test_stack_effect():
    for op in dis.opmap.values():
        for oparg in range(0, 15):
            for jump in (None, True, False):
                try:
                    ref = dis.stack_effect(op, oparg, jump=jump)
                except ValueError:
                    continue
                if op == 130 and oparg >= 3:  # RAISE_VARARGS
                    continue
                out = stack_effect(op, oparg, jump)
                assert ref == out[2] - out[
                    1], f"op: {dis.opname[op]}({op}), oparg: {oparg}, jump: {jump}, ref: {ref}, out: {out}"
