from frontend.instruction import ci
from frontend.bytecode_writter import update_offsets
from frontend.frame_saver import ProcessedCode
import pytest

num_nop = 2


def gen_processed_code(gen_inst_fn):
    original_insts = gen_inst_fn()
    nops = [ci("NOP") for _ in range(num_nop)]
    guarded_insts = [*nops, *gen_inst_fn()]
    update_offsets(original_insts)
    update_offsets(guarded_insts)

    for o, g in zip(original_insts, guarded_insts[num_nop:]):
        g.original_inst = o
    processed_code = ProcessedCode(original_insts, guarded_insts, [nops[-1]])
    return original_insts, guarded_insts, processed_code


def check_pc(gen_inst_fn, last_pcs, next_pcs):
    original_insts, guarded_insts, processed_code = gen_processed_code(
        gen_inst_fn)
    last_orig_pcs = [
        processed_code.get_orig_pc(inst.offset) for inst in guarded_insts
    ]
    assert last_orig_pcs == last_pcs
    next_orig_pcs = [
        processed_code.get_next_orig_pc(inst.offset) for inst in guarded_insts
    ]
    assert next_orig_pcs == next_pcs


def test_pc_simple_program():

    def gen_inst():
        return [
            ci("LOAD_CONST", 1),
            ci("LOAD_CONST", 2),
            ci("BINARY_ADD"),
            ci("RETURN_VALUE"),
        ]

    last_pcs = [-2, -1, 0, 1, 2, 3]
    next_pcs = [-1, -1, 1, 2, 3, 4]
    check_pc(gen_inst, last_pcs, next_pcs)


def test_pc_with_ext():

    def gen_inst():
        return [
            ci("LOAD_CONST", 1),
            ci("EXTENDED_ARG", 1),
            ci("EXTENDED_ARG", 1),
            ci("LOAD_CONST", 2),
            ci("RETURN_VALUE"),
        ]

    last_pcs = [-2, -1, 0, 3, 3, 3, 4]
    next_pcs = [-1, -1, 3, 4, 4, 4, 5]
    check_pc(gen_inst, last_pcs, next_pcs)


def test_pc_with_cond_jump_if_true():

    def gen_inst():
        insts = [
            ci("LOAD_CONST", 1),
            ci("LOAD_CONST", 2),
            ci("POP_JUMP_IF_TRUE", 1),
            ci("LOAD_CONST", 3),
            ci("RETURN_VALUE"),
        ]
        insts[2].target = insts[-1]
        return insts

    _, _, processed_code = gen_processed_code(gen_inst)
    with pytest.raises(ValueError):
        processed_code.get_next_orig_pc((2 + num_nop) * 2)
    assert processed_code.get_next_orig_pc((2 + num_nop) * 2, True) == 4
    assert processed_code.get_next_orig_pc((2 + num_nop) * 2, False) == 3


def test_pc_with_cond_jump_if_false():

    def gen_inst():
        insts = [
            ci("LOAD_CONST", 1),
            ci("LOAD_CONST", 2),
            ci("POP_JUMP_IF_FALSE", 1),
            ci("LOAD_CONST", 3),
            ci("RETURN_VALUE"),
        ]
        insts[2].target = insts[-1]
        return insts

    _, _, processed_code = gen_processed_code(gen_inst)
    with pytest.raises(ValueError):
        processed_code.get_next_orig_pc((2 + num_nop) * 2)
    assert processed_code.get_next_orig_pc((2 + num_nop) * 2, False) == 4
    assert processed_code.get_next_orig_pc((2 + num_nop) * 2, True) == 3


def test_code_with_force_jump():

    def gen_inst():
        insts = [
            ci("LOAD_CONST", 1),
            ci("LOAD_CONST", 2),
            ci("JUMP_ABSOLUTE", 1),
            ci("LOAD_CONST", 3),
            ci("RETURN_VALUE"),
        ]
        insts[2].target = insts[-1]
        return insts

    _, _, processed_code = gen_processed_code(gen_inst)
    assert processed_code.get_next_orig_pc((2 + num_nop) * 2) == 4
