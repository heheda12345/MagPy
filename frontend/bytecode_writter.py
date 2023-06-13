import dataclasses
from typing import Any, Dict, List
from frontend.bytecode_analysis import stacksize_analysis
from frontend.instruction import Instruction, convert_instruction, ci
import dis
import types
import sys
from typing import Tuple, Callable


def get_code_keys() -> List[str]:
    keys = ["co_argcount"]
    keys.append("co_posonlyargcount")
    keys.extend([
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
    ])
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
    keys.extend([
        "co_freevars",
        "co_cellvars",
    ])
    return keys


HAS_LOCAL = set(dis.haslocal)
HAS_NAME = set(dis.hasname)


# map from var name to index
def fix_vars(instructions: List[Instruction], code_options: Dict[str,
                                                                 Any]) -> None:
    varnames = {
        name: idx for idx, name in enumerate(code_options["co_varnames"])
    }
    names = {name: idx for idx, name in enumerate(code_options["co_names"])}
    for i in range(len(instructions)):
        if instructions[i].opcode in HAS_LOCAL:
            instructions[i].arg = varnames[instructions[i].argval]
        elif instructions[i].opcode in HAS_NAME:
            instructions[i].arg = names[instructions[i].argval]


def instruction_size(inst: Instruction) -> int:
    return 2


def update_offsets(instructions: List[Instruction]) -> None:
    offset = 0
    for inst in instructions:
        inst.offset = offset
        offset += instruction_size(inst)


def devirtualize_jumps(instructions: List[Instruction]) -> None:
    """Fill in args for virtualized jump target after instructions may have moved"""
    indexof = {id(inst): i for i, inst, in enumerate(instructions)}
    jumps = set(dis.hasjabs).union(set(dis.hasjrel))

    for inst in instructions:
        if inst.opcode in jumps:
            target = inst.target
            assert target is not None
            target_index = indexof[id(target)]
            for offset in (1, 2, 3):
                if (target_index >= offset and
                        instructions[target_index - offset].opcode
                        == dis.EXTENDED_ARG):
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
                    raise RuntimeError(
                        "Python 3.11+ should not have absolute jumps")
            else:  # relative jump
                # byte offset between target and next instruction
                assert target.offset is not None
                assert inst.offset is not None
                inst.arg = int(target.offset - inst.offset -
                               instruction_size(inst))
                if inst.arg < 0:
                    raise RuntimeError(
                        "Got negative jump offset for Python < 3.11")
            inst.argval = target.offset


def fix_extended_args(instructions: List[Instruction]) -> int:
    """Fill in correct argvals for EXTENDED_ARG ops"""
    output: List[Instruction] = []

    def maybe_pop_n(n: int) -> None:
        for _ in range(n):
            if output and output[-1].opcode == dis.EXTENDED_ARG:
                output.pop()

    for i, inst in enumerate(instructions):
        if inst.opcode == dis.EXTENDED_ARG:
            # Leave this instruction alone for now so we never shrink code
            inst.arg = 0
        elif inst.arg and inst.arg > 0xFFFFFF:
            maybe_pop_n(3)
            output.append(ci("EXTENDED_ARG", inst.arg >> 24))
            output.append(ci("EXTENDED_ARG", inst.arg >> 16))
            output.append(ci("EXTENDED_ARG", inst.arg >> 8))
        elif inst.arg and inst.arg > 0xFFFF:
            maybe_pop_n(2)
            output.append(ci("EXTENDED_ARG", inst.arg >> 16))
            output.append(ci("EXTENDED_ARG", inst.arg >> 8))
        elif inst.arg and inst.arg > 0xFF:
            maybe_pop_n(1)
            output.append(ci("EXTENDED_ARG", inst.arg >> 8))
        output.append(inst)

    added = len(output) - len(instructions)
    assert added >= 0
    instructions[:] = output
    return added


def remove_extra_line_nums(instructions: List[Instruction]) -> None:
    """Remove extra starts line properties before packing bytecode"""

    cur_line_no = None

    def remove_line_num(inst: Instruction) -> None:
        nonlocal cur_line_no
        if inst.starts_line is None:
            return
        elif inst.starts_line == cur_line_no:
            inst.starts_line = None
        else:
            cur_line_no = inst.starts_line

    for inst in instructions:
        remove_line_num(inst)


def lnotab_writer(
        lineno: int,
        byteno: int = 0) -> Tuple[List[int], Callable[[int, int], None]]:
    """
    Used to create typing.CodeType.co_lnotab
    See https://github.com/python/cpython/blob/main/Objects/lnotab_notes.txt
    This is the internal format of the line number table if Python < 3.10
    """
    assert sys.version_info < (3, 10)
    lnotab: List[int] = []

    def update(lineno_new: int, byteno_new: int) -> None:
        nonlocal byteno, lineno
        while byteno_new != byteno or lineno_new != lineno:
            byte_offset = max(0, min(byteno_new - byteno, 255))
            line_offset = max(-128, min(lineno_new - lineno, 127))
            assert byte_offset != 0 or line_offset != 0
            byteno += byte_offset
            lineno += line_offset
            lnotab.extend((byte_offset, line_offset & 0xFF))

    return lnotab, update


def assemble(instructions: List[Instruction],
             firstlineno: int) -> Tuple[bytes, bytes]:
    """Do the opposite of dis.get_instructions()"""
    code: List[int] = []
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


def add_name_to_code_options(code_options: Dict[str, Any]) -> None:
    code_options["co_names"] = (*code_options["co_names"], "fake_print")


def fix_constants(instructions: List[Instruction],
                  code_options: Dict[str, Any]) -> None:
    const_set = set(code_options["co_consts"])
    const_list = list(code_options["co_consts"])
    LOAD_CONST = dis.opmap["LOAD_CONST"]
    for inst in instructions:
        if inst.opcode == LOAD_CONST and inst.argval not in const_set:
            const_list.append(inst.argval)
            inst.arg = len(const_list) - 1
    code_options["co_consts"] = tuple(const_list)


def assemble_instructions(
        instructions: List[Instruction],
        code_options: Dict[str,
                           Any]) -> Tuple[List[Instruction], types.CodeType]:
    add_name_to_code_options(code_options)
    fix_vars(instructions, code_options)
    fix_constants(instructions, code_options)
    keys = get_code_keys()
    dirty = True
    while dirty:
        update_offsets(instructions)
        devirtualize_jumps(instructions)
        dirty = fix_extended_args(instructions) > 0
    remove_extra_line_nums(instructions)

    bytecode, lnotab = assemble(instructions, code_options["co_firstlineno"])
    if sys.version_info < (3, 10):
        code_options["co_lnotab"] = lnotab
    else:
        code_options["co_linetable"] = lnotab
    code_options["co_code"] = bytecode
    code_options["co_nlocals"] = len(code_options["co_varnames"])
    code_options["co_stacksize"] = stacksize_analysis(instructions)
    assert set(keys) - {"co_posonlyargcount"} == set(
        code_options.keys()) - {"co_posonlyargcount"}
    if sys.version_info >= (3, 11):
        # generated code doesn't contain exceptions, so leave exception table empty
        code_options["co_exceptiontable"] = b""
    code = types.CodeType(*[code_options[k] for k in keys])
    return instructions, code


def virtualize_jumps(instructions: List[Instruction]) -> None:
    """Replace jump targets with pointers to make editing easier"""
    jump_targets = {inst.offset: inst for inst in instructions}

    for inst in instructions:
        if inst.opcode in dis.hasjabs or inst.opcode in dis.hasjrel:
            for offset in (0, 2, 4, 6):
                if jump_targets[inst.argval +
                                offset].opcode != dis.EXTENDED_ARG:
                    inst.target = jump_targets[inst.argval + offset]
                    break


def strip_extended_args(instructions: List[Instruction]) -> None:
    instructions[:] = [i for i in instructions if i.opcode != dis.EXTENDED_ARG]


def get_instructions(code: types.CodeType) -> List[Instruction]:
    instructions = dis.Bytecode(code)
    instructions_converted = [convert_instruction(i) for i in instructions]
    virtualize_jumps(instructions_converted)
    strip_extended_args(instructions_converted)
    return instructions_converted


def add_guard(instructions: List[Instruction], start_inst: int, end_inst: int,
              frame_id: int, callsite_id: int,
              call_graph_insts: List[Instruction], call_fn_num_args: int,
              recover_stack_insts: List[Instruction]) -> None:
    guard_code = [
        ci("LOAD_GLOBAL", "guard_match"),
        ci("LOAD_CONST", frame_id),
        ci("LOAD_CONST", callsite_id),
        ci("LOAD_GLOBAL", "locals"),
        ci("CALL_FUNCTION", 0),
        ci("CALL_FUNCTION", 3),
        ci("STORE_FAST", "__graph_fn"),
        ci("LOAD_GLOBAL", "callable"),
        ci("LOAD_FAST", "__graph_fn"),
        ci("CALL_FUNCTION", 1),
        ci("POP_JUMP_IF_FALSE", target=instructions[start_inst]),
        ci("LOAD_FAST", "__graph_fn"),
        *call_graph_insts,
        ci("CALL_FUNCTION", call_fn_num_args),
        *recover_stack_insts,
        ci("JUMP_FORWARD", target=instructions[end_inst]),
    ]
    instructions[start_inst:start_inst] = guard_code


def add_name(code_options: Dict[str, Any], varnames: List[str],
             names: List[str]) -> None:
    code_options["co_varnames"] = (*code_options["co_varnames"],
                                   *tuple(varnames))
    code_options["co_names"] = (*code_options["co_names"], *tuple(names))
    code_options["co_nlocals"] = len(code_options["co_varnames"])


def rewrite_bytecode(code: types.CodeType) -> types.CodeType:
    instructions = get_instructions(code)
    for i, inst in enumerate(instructions):
        print(i, inst, id(inst), id(inst.target))
    add_guard(instructions, 0, 7, 0, 0, [], 0, [])
    print("guarded code")
    for i, inst in enumerate(instructions):
        print(i, inst, id(inst), id(inst.target))
    keys = get_code_keys()
    code_options = {k: getattr(code, k) for k in keys}
    add_name(code_options, ["__graph_fn"],
             ["guard_match", "locals", "callable"])
    code_options["co_stacksize"] += 4
    new_code = assemble_instructions(instructions, code_options)[1]
    return new_code


# test code


def add_print_to_return(code: types.CodeType) -> types.CodeType:
    instructions = get_instructions(code)
    for i, inst in enumerate(instructions):
        if inst.opcode == dis.opmap["RETURN_VALUE"]:
            new_insts = [
                ci("DUP_TOP"),
                ci("LOAD_GLOBAL", "print"),
                ci("ROT_TWO"),
                ci("LOAD_CONST", None, "print: return value is"),
                ci("ROT_TWO"),
                ci("CALL_FUNCTION", 2),
                ci("POP_TOP")
            ]
            for j, new_inst in enumerate(new_insts):
                instructions.insert(i + j, new_inst)
            break
    keys = get_code_keys()
    code_options = {k: getattr(code, k) for k in keys}
    new_code = assemble_instructions(instructions, code_options)[1]
    return new_code
