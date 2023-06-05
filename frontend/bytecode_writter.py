import dataclasses
from typing import Any, Dict, List, Optional, Tuple
from frontend.bytecode_analysis import stacksize_analysis
import dis
import types
import sys

@dataclasses.dataclass
class Instruction:
    """A mutable version of dis.Instruction"""

    opcode: int
    opname: str
    arg: Optional[int]
    argval: Any
    offset: Optional[int] = None
    starts_line: Optional[int] = None
    is_jump_target: bool = False
    # extra fields to make modification easier:
    target: Optional["Instruction"] = None

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return id(self) == id(other)


def convert_instruction(i: dis.Instruction):
    return Instruction(
        i.opcode,
        i.opname,
        i.arg,
        i.argval,
        i.offset,
        i.starts_line,
        i.is_jump_target,
    )


class _NotProvided:
    pass

def create_instruction(name, arg=None, argval=_NotProvided, target=None):
    if argval is _NotProvided:
        argval = arg
    return Instruction(
        opcode=dis.opmap[name], opname=name, arg=arg, argval=argval, target=target
    )


def get_code_keys():
    keys = ["co_argcount"]
    keys.append("co_posonlyargcount")
    keys.extend(
        [
            "co_kwonlyargcount",
            "co_nlocals",
            "co_stacksize",
            "co_flags",
            "co_code",
            "co_consts",
            "co_names",
            "co_varnames",
            "co_filename",
            "co_name",
        ]
    )
    if sys.version_info >= (3, 11):
        keys.append("co_qualname")
    keys.append("co_firstlineno")
    if sys.version_info >= (3, 10):
        keys.append("co_linetable")
    else:
        keys.append("co_lnotab")
    if sys.version_info >= (3, 11):
        # not documented, but introduced in https://github.com/python/cpython/issues/84403
        keys.append("co_exceptiontable")
    keys.extend(
        [
            "co_freevars",
            "co_cellvars",
        ]
    )
    return keys


HAS_LOCAL = set(dis.haslocal)
HAS_NAME = set(dis.hasname)


# map from var name to index
def fix_vars(instructions: List[Instruction], code_options):
    print("co_names:", code_options["co_names"])
    varnames = {name: idx for idx, name in enumerate(code_options["co_varnames"])}
    names = {name: idx for idx, name in enumerate(code_options["co_names"])}
    for i in range(len(instructions)):
        if instructions[i].opcode in HAS_LOCAL:
            instructions[i].arg = varnames[instructions[i].argval]
        elif instructions[i].opcode in HAS_NAME:
            instructions[i].arg = names[instructions[i].argval]


def instruction_size(inst):
    return 2


def update_offsets(instructions):
    offset = 0
    for inst in instructions:
        inst.offset = offset
        offset += instruction_size(inst)


def devirtualize_jumps(instructions):
    """Fill in args for virtualized jump target after instructions may have moved"""
    indexof = {id(inst): i for i, inst, in enumerate(instructions)}
    jumps = set(dis.hasjabs).union(set(dis.hasjrel))

    for inst in instructions:
        if inst.opcode in jumps:
            target = inst.target
            target_index = indexof[id(target)]
            for offset in (1, 2, 3):
                if (
                    target_index >= offset
                    and instructions[target_index - offset].opcode == dis.EXTENDED_ARG
                ):
                    target = instructions[target_index - offset]
                else:
                    break

            if inst.opcode in dis.hasjabs:
                if sys.version_info < (3, 10):
                    inst.arg = target.offset
                elif sys.version_info < (3, 11):
                    # `arg` is expected to be bytecode offset, whereas `offset` is byte offset.
                    # Divide since bytecode is 2 bytes large.
                    inst.arg = int(target.offset / 2)
                else:
                    raise RuntimeError("Python 3.11+ should not have absolute jumps")
            else:  # relative jump
                # byte offset between target and next instruction
                inst.arg = int(target.offset - inst.offset - instruction_size(inst))
                if inst.arg < 0:
                    if sys.version_info < (3, 11):
                        raise RuntimeError("Got negative jump offset for Python < 3.11")
                    inst.arg = -inst.arg
                    # forward jumps become backward
                    if "FORWARD" in inst.opname:
                        flip_jump_direction(inst)
                elif inst.arg > 0:
                    # backward jumps become forward
                    if sys.version_info >= (3, 11) and "BACKWARD" in inst.opname:
                        flip_jump_direction(inst)
                if sys.version_info >= (3, 10):
                    # see bytecode size comment in the absolute jump case above
                    inst.arg //= 2
            inst.argval = target.offset
            inst.argrepr = f"to {target.offset}"


def fix_extended_args(instructions: List[Instruction]):
    """Fill in correct argvals for EXTENDED_ARG ops"""
    output = []

    def maybe_pop_n(n):
        for _ in range(n):
            if output and output[-1].opcode == dis.EXTENDED_ARG:
                output.pop()

    for i, inst in enumerate(instructions):
        if inst.opcode == dis.EXTENDED_ARG:
            # Leave this instruction alone for now so we never shrink code
            inst.arg = 0
        elif inst.arg and inst.arg > 0xFFFFFF:
            maybe_pop_n(3)
            output.append(create_instruction("EXTENDED_ARG", inst.arg >> 24))
            output.append(create_instruction("EXTENDED_ARG", inst.arg >> 16))
            output.append(create_instruction("EXTENDED_ARG", inst.arg >> 8))
        elif inst.arg and inst.arg > 0xFFFF:
            maybe_pop_n(2)
            output.append(create_instruction("EXTENDED_ARG", inst.arg >> 16))
            output.append(create_instruction("EXTENDED_ARG", inst.arg >> 8))
        elif inst.arg and inst.arg > 0xFF:
            maybe_pop_n(1)
            output.append(create_instruction("EXTENDED_ARG", inst.arg >> 8))
        output.append(inst)

    added = len(output) - len(instructions)
    assert added >= 0
    instructions[:] = output
    return added


def remove_extra_line_nums(instructions):
    """Remove extra starts line properties before packing bytecode"""

    cur_line_no = None

    def remove_line_num(inst):
        nonlocal cur_line_no
        if inst.starts_line is None:
            return
        elif inst.starts_line == cur_line_no:
            inst.starts_line = None
        else:
            cur_line_no = inst.starts_line

    for inst in instructions:
        remove_line_num(inst)


def lnotab_writer(lineno, byteno=0):
    """
    Used to create typing.CodeType.co_lnotab
    See https://github.com/python/cpython/blob/main/Objects/lnotab_notes.txt
    This is the internal format of the line number table if Python < 3.10
    """
    assert sys.version_info < (3, 10)
    lnotab = []

    def update(lineno_new, byteno_new):
        nonlocal byteno, lineno
        while byteno_new != byteno or lineno_new != lineno:
            byte_offset = max(0, min(byteno_new - byteno, 255))
            line_offset = max(-128, min(lineno_new - lineno, 127))
            assert byte_offset != 0 or line_offset != 0
            byteno += byte_offset
            lineno += line_offset
            lnotab.extend((byte_offset, line_offset & 0xFF))

    return lnotab, update


def assemble(instructions: List[Instruction], firstlineno):
    """Do the opposite of dis.get_instructions()"""
    code = []
    lnotab, update_lineno = lnotab_writer(firstlineno)

    for inst in instructions:
        if inst.starts_line is not None:
            update_lineno(inst.starts_line, len(code))
        arg = inst.arg or 0
        code.extend((inst.opcode, arg & 0xFF))
        if sys.version_info >= (3, 11):
            for _ in range(instruction_size(inst) // 2 - 1):
                code.extend((0, 0))

    if sys.version_info >= (3, 10):
        end(len(code))

    return bytes(code), bytes(lnotab)


def assemble_instructions(instructions: List[Instruction], code_options) -> types.CodeType:
    code_options["co_names"] = (*code_options["co_names"], "fake_print")
    fix_vars(instructions, code_options)
    keys = get_code_keys()
    dirty = True
    while dirty:
        update_offsets(instructions)
        devirtualize_jumps(instructions)
        dirty = fix_extended_args(instructions)
    remove_extra_line_nums(instructions)

    bytecode, lnotab = assemble(instructions, code_options["co_firstlineno"])
    if sys.version_info < (3, 10):
        code_options["co_lnotab"] = lnotab
    else:
        code_options["co_linetable"] = lnotab
    code_options["co_code"] = bytecode
    code_options["co_nlocals"] = len(code_options["co_varnames"])
    code_options["co_stacksize"] = stacksize_analysis(instructions)
    assert set(keys) - {"co_posonlyargcount"} == set(code_options.keys()) - {
        "co_posonlyargcount"
    }
    if sys.version_info >= (3, 11):
        # generated code doesn't contain exceptions, so leave exception table empty
        code_options["co_exceptiontable"] = b""
    return instructions, types.CodeType(*[code_options[k] for k in keys])


def virtualize_jumps(instructions):
    """Replace jump targets with pointers to make editing easier"""
    jump_targets = {inst.offset: inst for inst in instructions}

    for inst in instructions:
        if inst.opcode in dis.hasjabs or inst.opcode in dis.hasjrel:
            for offset in (0, 2, 4, 6):
                if jump_targets[inst.argval + offset].opcode != dis.EXTENDED_ARG:
                    inst.target = jump_targets[inst.argval + offset]
                    break


def strip_extended_args(instructions: List[Instruction]):
    instructions[:] = [i for i in instructions if i.opcode != dis.EXTENDED_ARG]


def get_instructions(code: types.CodeType) -> List[Instruction]:
    instructions = dis.Bytecode(code)
    instructions = [convert_instruction(i) for i in instructions]
    virtualize_jumps(instructions)
    strip_extended_args(instructions)
    return instructions


# test code

def add_print_to_return(code: types.CodeType) -> List[Instruction]:
    instructions = get_instructions(code)
    old_const_count = len(code.co_consts)
    for i, inst in enumerate(instructions):
        if inst.opcode == dis.opmap["RETURN_VALUE"]:
            new_insts = [
                create_instruction("DUP_TOP"),
                create_instruction("LOAD_GLOBAL", "fake_print"),
                create_instruction("ROT_TWO"),
                create_instruction("LOAD_CONST", old_const_count, "print: return value is"),
                create_instruction("ROT_TWO"),
                create_instruction("CALL_FUNCTION", 2),
                create_instruction("POP_TOP")
            ]
            for j, new_inst in enumerate(new_insts):
                instructions.insert(i + j, new_inst)
            break
    keys = get_code_keys()
    code_options = {k: getattr(code, k) for k in keys}
    code_options["co_consts"] = (*code.co_consts, "return value is")
    new_code = assemble_instructions(instructions, code_options)[1]
    return new_code

import inspect

if __name__ == '__main__':
    def test():
        test_func_frame = inspect.currentframe().f_back
        code = test_func_frame.f_code
        insts = get_instructions(code)
        for inst in insts:
            print(inst)
        print("=====================")
        insts.insert(3, Instruction(116, 'LOAD_GLOBAL', 0, 'print'))
        insts.insert(4, Instruction(100, 'LOAD_CONST', 2, 888))
        insts.insert(5, Instruction(131, 'CALL_FUNCTION', 1, 1))
        insts.insert(6, Instruction(100, 'LOAD_CONST', 0, None))
        insts.insert(7, Instruction(1, 'POP_TOP', None, None))
        insts.insert(8, Instruction(83, 'RETURN_VALUE', None, None))
        for inst in insts:
            print(inst)
        keys = get_code_keys()
        code_options = {k: getattr(code, k) for k in keys}
        code_options["co_consts"] = (None, 666, 888)
        assert len(code_options["co_varnames"]) == code_options["co_nlocals"]
        new_code = assemble_instructions(insts, code_options)[1]
        exec(new_code, {}, {})


    def test_func():
        print(666)
        test()
        

    test_func()