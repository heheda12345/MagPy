import dataclasses
from typing import Any, Dict, List
import dis
import types
import sys
from typing import Tuple, Callable
import copy
import itertools
from .bytecode_analysis import stacksize_analysis, end_of_control_flow, livevars_analysis, eliminate_dead_code, jump_only_opnames
from .instruction import Instruction, convert_instruction, ci, format_insts
from .code import generate_code_map, ProcessedCode
from .cache import get_frame_cache, CachedGraph
from .store_pos import StorePos, StoreInStack, StoreInLocal
from .pycode_writer import new_name
from .dynamic import need_branch_rewrite, get_branch_rewrite_pcs


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
HAS_FREE = set(dis.hasfree)


# map from var name to index
def fix_vars(instructions: List[Instruction], code_options: Dict[str,
                                                                 Any]) -> None:
    varnames = {
        name: idx for idx, name in enumerate(code_options["co_varnames"])
    }
    names = {name: idx for idx, name in enumerate(code_options["co_names"])}
    freenames = {
        name: idx for idx, name in enumerate(
            itertools.chain(code_options["co_cellvars"],
                            code_options["co_freevars"]))
    }
    for i in range(len(instructions)):
        if instructions[i].opcode in HAS_LOCAL:
            instructions[i].arg = varnames[instructions[i].argval]
        elif instructions[i].opcode in HAS_NAME:
            instructions[i].arg = names[instructions[i].argval]
        elif instructions[i].opcode in HAS_FREE:
            instructions[i].arg = freenames[instructions[i].argval]


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

    for i, inst in enumerate(instructions):
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
        # print("inst", inst)
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


def fix_constants(instructions: List[Instruction],
                  code_options: Dict[str, Any]) -> None:
    # use type as a key because 1 == 1.0, so python3 -c "a = set([1]); print(1.0 in a)" returns True
    const_set = set({(type(x), x) for x in code_options["co_consts"]})
    const_list = list({(type(x), x) for x in code_options["co_consts"]})
    LOAD_CONST = dis.opmap["LOAD_CONST"]
    for inst in instructions:
        if inst.opcode == LOAD_CONST:
            entry = (type(inst.argval), inst.argval)
            if entry not in const_set:
                const_list.append(entry)
                const_set.add(entry)
            inst.arg = const_list.index(entry)
    print("const_list", const_list)
    code_options["co_consts"] = tuple((x[1] for x in const_list))


def fix_instructions_for_assemble(instructions: List[Instruction],
                                  code_options: Dict[str, Any]) -> None:
    add_name(instructions, code_options)
    strip_extended_args(instructions)
    fix_vars(instructions, code_options)
    fix_constants(instructions, code_options)
    dirty = True
    update_offsets(instructions)
    while dirty:
        devirtualize_jumps(instructions)
        dirty = fix_extended_args(instructions) > 0
        update_offsets(instructions)
    remove_extra_line_nums(instructions)


def assemble_instructions(
        instructions: List[Instruction],
        code_options: Dict[str,
                           Any]) -> Tuple[List[Instruction], types.CodeType]:
    keys = get_code_keys()
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
    return instructions_converted


def add_callsite(orignal_insts: List[Instruction], is_callee: bool,
                 cached_graphs: List[CachedGraph], frame_id: int,
                 callsite_id: int,
                 start_pc: int) -> tuple[list[Instruction], list[Instruction]]:
    assert orignal_insts[start_pc].opname != "RETURN_VALUE"
    in_trace_insts = []
    disable_trace_insts = []
    if start_pc != 0:
        disable_trace_insts.extend([
            ci("LOAD_GLOBAL", "disable_trace"),
            ci("LOAD_CONST", frame_id),
            ci("CALL_FUNCTION", 1),
            ci("POP_TOP"),
        ])
        in_trace_insts.extend(disable_trace_insts[:-1])

    start_stack_size = cached_graphs[0].start_stack_size if cached_graphs else 0
    for graph in cached_graphs:
        assert graph.start_stack_size == start_stack_size

    prepare_stack_insts = [
        ci("STORE_FAST", f"__stack__{i}") for i in range(start_stack_size)
    ]
    call_guard_insts = [
        *prepare_stack_insts,
        ci("LOAD_GLOBAL", "guard_match"),
        ci("LOAD_CONST", frame_id),
        ci("LOAD_CONST", callsite_id),
        ci("LOAD_GLOBAL", "locals"),
        ci("CALL_FUNCTION", 0),
        ci("CALL_FUNCTION",
           3,
           comment="call guard_match(frame_id, callsite_id, locals)"),
        ci("UNPACK_SEQUENCE", 2),
        ci("STORE_FAST", "__case_idx"),
        ci("STORE_FAST", "__graph_fn"),
    ]
    possible_matches: list[list[Instruction]] = []
    for i, graph in enumerate(cached_graphs):
        insts = [
            ci("LOAD_FAST", "__case_idx"),
            ci("LOAD_CONST", i),
            ci("COMPARE_OP", dis.cmp_op.index("=="), "=="),
            ci("POP_JUMP_IF_FALSE", target=None),
            ci("LOAD_FAST", "__graph_fn"),
            ci("LOAD_GLOBAL", "locals"),
            ci("CALL_FUNCTION", 0),
            ci("CALL_FUNCTION", 1, comment=f"call graph_fn (key={graph.key})"),
        ]
        if len(graph.return_values) == 0:
            insts.append(ci("POP_TOP"))
        elif len(graph.return_values) == 1:
            pass
        elif len(graph.return_values) > 1:
            insts.append(ci("UNPACK_SEQUENCE", len(graph.return_values)))
        stack_var_start = False
        for return_value in graph.return_values:
            if isinstance(return_value, StoreInStack):
                stack_var_start = True
                continue
            else:
                assert not stack_var_start
            if isinstance(return_value, StoreInLocal):
                insts.append(ci("STORE_FAST", return_value.name))
            else:
                raise NotImplementedError(
                    f"{return_value} is not supported in bytecode writer")
        if orignal_insts[graph.end_pc].opname != 'RETURN_VALUE':
            insts.extend([
                ci("LOAD_GLOBAL", "enable_trace"),
                ci("LOAD_CONST", frame_id),
                ci("CALL_FUNCTION", 1),
                ci("POP_TOP"),
                ci("JUMP_ABSOLUTE", target=orignal_insts[graph.end_pc])
            ])
            in_trace_insts.extend(insts[-2:])
        else:
            if is_callee:
                insts.extend([
                    ci("LOAD_GLOBAL", "enable_trace"),
                    ci("LOAD_CONST", frame_id),
                    ci("CALL_FUNCTION", 1),
                    ci("POP_TOP"),
                    ci("RETURN_VALUE")
                ])
                in_trace_insts.extend(insts[-3:])
            else:
                insts.append(ci("RETURN_VALUE"))
        possible_matches.append(insts)
    restore_stack_insts = [
        ci("LOAD_FAST", f"__stack__{i}")
        for i in range(start_stack_size - 1, -1, -1)
    ]
    nomatch_code = [
        *restore_stack_insts,
        ci("LOAD_GLOBAL", "enable_trace"),
        ci("LOAD_CONST", frame_id),
        ci("CALL_FUNCTION", 1),
        ci("POP_TOP"),
        ci("JUMP_ABSOLUTE", target=orignal_insts[start_pc]),
    ]
    possible_matches.append(nomatch_code)
    in_trace_insts.extend(nomatch_code[-2:])
    for insts1, insts2 in zip(possible_matches[:-1], possible_matches[1:]):
        assert insts1[3].opname == "POP_JUMP_IF_FALSE"
        insts1[3].target = insts2[0]
    match_and_run_insts = []
    for insts in possible_matches:
        match_and_run_insts.extend(insts)
    callsite_insts = [
        *disable_trace_insts,
        *call_guard_insts,
        *match_and_run_insts,
    ]
    return callsite_insts, in_trace_insts


def add_name(instructions: list[Instruction], code_options: Dict[str,
                                                                 Any]) -> None:
    co_varnames = list(code_options["co_varnames"])
    co_varnames_set = set(co_varnames)
    co_names = list(code_options["co_names"])
    co_names_set = set(co_names)
    co_cellvars = list(code_options["co_cellvars"]) + list(
        code_options["co_freevars"])  # BUG: should put freevars behind cellvars
    co_cellvars_set = set(co_cellvars)
    co_freevars_set = set(code_options["co_freevars"])
    for inst in instructions:
        if inst.opcode in HAS_LOCAL:
            if inst.argval not in co_varnames_set:
                co_varnames.append(inst.argval)
                co_varnames_set.add(inst.argval)
        if inst.opcode in HAS_NAME:
            if inst.argval not in co_names_set:
                co_names.append(inst.argval)
                co_names_set.add(inst.argval)
        if inst.opcode in HAS_FREE:
            if inst.argval not in co_cellvars_set:
                co_cellvars.append(inst.argval)
                co_cellvars_set.add(inst.argval)
    code_options["co_varnames"] = tuple(
        co_varnames[:code_options["co_argcount"]] + [
            x for x in co_varnames[code_options["co_argcount"]:]
            if x not in co_cellvars_set
        ])
    code_options["co_names"] = tuple(co_names)
    code_options["co_cellvars"] = tuple(
        [x for x in co_cellvars if x not in co_freevars_set])
    code_options["co_nlocals"] = len(co_varnames)


SHOULD_NOT_CALL_REWRITE: bool = False  # for testing


def rewrite_branch(original_instructions: list[Instruction],
                   original_code: types.CodeType,
                   dynamic_pcs: list[int]) -> list[Instruction]:
    instructions = copy.deepcopy(original_instructions)
    virtualize_jumps(instructions)
    if len(dynamic_pcs) != 1:
        raise NotImplementedError
    dynamic_pc = dynamic_pcs[0]
    end_pc = end_of_control_flow(instructions, dynamic_pc)
    jump_target = instructions[dynamic_pc].target
    assert jump_target is not None
    branches = [
        eliminate_dead_code(instructions, dynamic_pc + 1, [end_pc]),
        eliminate_dead_code(instructions, jump_target, [end_pc])
    ]
    live_vars = livevars_analysis(instructions, instructions[end_pc])
    create_func_insts = []
    new_local_names = []
    local_fn_names = []
    need_deref_names: dict[str, str] = {}  # original name -> __local__xxxx
    for i, branch in enumerate(branches):
        binsts = list(instructions[x] for x in branch if x != end_pc)
        for inst in binsts:
            if inst.target is not None and inst.opname not in jump_only_opnames:
                raise NotImplementedError
        non_local_vars: dict[str, str] = {}
        local_vars: set[str] = set()
        for inst in binsts:
            if inst.opname == "LOAD_FAST" and inst.argval not in local_vars:
                if inst.argval not in non_local_vars:
                    if inst.argval not in need_deref_names:
                        need_deref_names[inst.argval] = new_name('__local__' +
                                                                 inst.argval)
                    non_local_vars[inst.argval] = need_deref_names[inst.argval]
            if inst.opname == "STORE_FAST":
                if inst.argval in live_vars:
                    if inst.argval not in non_local_vars:
                        if inst.argval not in need_deref_names:
                            need_deref_names[inst.argval] = new_name(
                                '__local__' + inst.argval)
                        non_local_vars[inst.argval] = need_deref_names[
                            inst.argval]
                else:
                    local_vars.add(inst.argval)
        local_list = tuple(local_vars)
        non_local_list = tuple(non_local_vars.values())
        # need_deref_names.update(non_local_vars)
        for inst in binsts:
            if inst.opname == "LOAD_FAST" and inst.argval in non_local_vars:
                inst.opname = "LOAD_DEREF"
                inst.opcode = dis.opmap["LOAD_DEREF"]
                inst.argval = non_local_vars[inst.argval]
                inst.arg = non_local_list.index(inst.argval)
            if inst.opname == "STORE_FAST" and inst.argval in non_local_vars:
                inst.opname = "STORE_DEREF"
                inst.opcode = dis.opmap["STORE_DEREF"]
                inst.argval = non_local_vars[inst.argval]
                inst.arg = non_local_list.index(inst.argval)
        if binsts[-1].opname in jump_only_opnames and binsts[
                -1].target == instructions[end_pc]:
            binsts[-1] = ci("LOAD_CONST", None)
            binsts.append(ci("RETURN_VALUE"))
        else:
            raise NotImplementedError
        co_name = new_name(f"__branch{i}")
        code_options: dict[str, Any] = {
            "co_argcount": 0,
            "co_cellvars": (),
            # "co_code": co_code,
            "co_consts": copy.deepcopy(original_code.co_consts),
            "co_filename": copy.deepcopy(original_code.co_filename),
            "co_firstlineno": 0,
            "co_flags": 0x13,  # CO_OPTIMIZED | CO_NEWLOCALS | CO_NESTED
            "co_freevars": non_local_list,
            "co_kwonlyargcount": 0,
            # "co_lnotab": co_lnotab,
            "co_name": co_name,
            "co_names": copy.deepcopy(original_code.co_names),
            "co_nlocals": len(local_list),
            "co_posonlyargcount": 0,
            # "co_stacksize": co_stacksize,
            "co_varnames": local_list,
        }
        fix_instructions_for_assemble(binsts, code_options)
        print(format_insts(binsts, allow_unknown_target=True))
        for k, v in code_options.items():
            print(k, v)
        _, new_code = assemble_instructions(binsts, code_options)
        create_func_insts.extend([
            ci("LOAD_CLOSURE", -1, nonlocal_name)
            for _, nonlocal_name in enumerate(non_local_list)
        ])
        local_fn_name = original_code.co_name + ".<locals>." + co_name
        create_func_insts.extend([
            ci("BUILD_TUPLE", len(non_local_list)),
            ci("LOAD_CONST", None, new_code),
            ci("LOAD_CONST", None, local_fn_name),
            ci("MAKE_FUNCTION", 0x8),
            ci("STORE_FAST", co_name)
        ])
        local_fn_names.append(co_name)

    if instructions[dynamic_pc].opname == "POP_JUMP_IF_TRUE":
        local_fn_names = [local_fn_names[1], local_fn_names[0]]
    elif instructions[dynamic_pc].opname == "POP_JUMP_IF_FALSE":
        pass
    else:
        raise NotImplementedError

    instructions = create_func_insts + instructions
    dynamic_pc = dynamic_pc + len(create_func_insts)
    end_pc = end_pc + len(create_func_insts)
    new_local_names.append(co_name)
    call_sys_cond_insts = [
        ci("LOAD_GLOBAL", "_frontend_compile_if_stmt"),
        ci("ROT_TWO"),
        ci("LOAD_FAST", local_fn_names[0]),
        ci("LOAD_FAST", local_fn_names[1]),
        ci("CALL_FUNCTION", 3),
        ci("POP_TOP"),
        ci("JUMP_ABSOLUTE", target=instructions[end_pc])
    ]
    instructions = instructions[:
                                dynamic_pc] + call_sys_cond_insts + instructions[
                                    dynamic_pc + 1:]
    live_pcs = eliminate_dead_code(instructions)
    instructions = [instructions[i] for i in live_pcs]
    keys = get_code_keys()
    code_options = {k: getattr(original_code, k) for k in keys}
    for inst in instructions:
        if inst.opname == "LOAD_FAST" and inst.argval in need_deref_names:
            inst.opname = "LOAD_DEREF"
            inst.opcode = dis.opmap["LOAD_DEREF"]
            inst.argval = need_deref_names[inst.argval]
        elif inst.opname == "STORE_FAST" and inst.argval in need_deref_names:
            inst.opname = "STORE_DEREF"
            inst.opcode = dis.opmap["STORE_DEREF"]
            inst.argval = need_deref_names[inst.argval]
    make_cell_arg_insts = []
    for name_old in code_options["co_varnames"][:code_options["co_argcount"]]:
        if name_old in need_deref_names:
            name_new = need_deref_names[name_old]
            make_cell_arg_insts.extend(
                [ci("LOAD_FAST", name_old),
                 ci("STORE_DEREF", name_new)])

    instructions = make_cell_arg_insts + instructions
    fix_instructions_for_assemble(instructions, code_options)
    # exit(0)
    return instructions, code_options


def rewrite_bytecode(code: types.CodeType, frame_id: int,
                     is_callee: bool) -> tuple[types.CodeType, ProcessedCode]:
    if SHOULD_NOT_CALL_REWRITE:
        raise RuntimeError("should not call rewrite_bytecode")
    original_instructions = get_instructions(code)
    if need_branch_rewrite(frame_id):
        original_instructions, code_options = rewrite_branch(
            original_instructions, code, get_branch_rewrite_pcs(frame_id))
    else:
        keys = get_code_keys()
        code_options = {k: getattr(code, k) for k in keys}
    instructions = copy.deepcopy(original_instructions)
    virtualize_jumps(instructions)
    for original_inst, inst in zip(original_instructions, instructions):
        inst.original_inst = original_inst
    instructions[0].is_start = True
    # print(format_insts(instructions))
    frame_cache = get_frame_cache(frame_id)
    # list of (start_pc, traced_instructions)
    run_traced_insts: list[tuple[int, list[Instruction]]] = []
    in_trace_insts = []
    if is_callee:
        final_insts = [
            ci("RETURN_VALUE"),
        ]
    else:
        final_insts = [
            ci("LOAD_GLOBAL", "disable_trace"),
            ci("LOAD_CONST", frame_id),
            ci("CALL_FUNCTION", 1),
            ci("POP_TOP"),
            ci("RETURN_VALUE")
        ]
    in_trace_insts.extend(final_insts)
    # new_names_all: dict[str, set[str]] = {"varnames": set(), "names": set()}
    for start_pc, callsite_id in frame_cache.callsite_id.items():
        cached_graphs = frame_cache.cached_graphs[start_pc]
        callsite_code, new_in_trace_insts = add_callsite(
            instructions, is_callee, cached_graphs, frame_id, callsite_id,
            start_pc)
        run_traced_insts.append((start_pc, callsite_code))
        in_trace_insts.extend(new_in_trace_insts)
        # new_names_all["varnames"].update(new_names["varnames"])
        # new_names_all["names"].update(new_names["names"])
    next_original_pc: list[tuple[Instruction, Instruction]] = []
    for i, inst in enumerate(instructions):
        if inst.opname == "RETURN_VALUE":
            original = inst.original_inst
            assert original is not None
            instructions[i] = ci("JUMP_ABSOLUTE", target=final_insts[0])
            instructions[i].is_end = True
            next_original_pc.append((original, instructions[i]))
            in_trace_insts.append(instructions[i])
            for jump_inst in instructions:
                if jump_inst.target is not None and jump_inst.target == inst:
                    jump_inst.target = instructions[i]
    run_traced_insts.sort(key=lambda x: x[0], reverse=True)
    for start_pc, traced_code in run_traced_insts:
        jump_inst = ci("JUMP_ABSOLUTE", target=traced_code[0])
        next_original_pc.append((original_instructions[start_pc], jump_inst))
        instructions.insert(start_pc, jump_inst)
        in_trace_insts.append(jump_inst)
    run_traced_insts.reverse()
    for start_pc, traced_code in run_traced_insts:
        instructions.extend(traced_code)
    if is_callee:
        disable_trace_at_start = [
            ci("LOAD_GLOBAL", "disable_trace"),
            ci("LOAD_CONST", frame_id),
            ci("CALL_FUNCTION", 1),
            ci("POP_TOP"),
        ]
        in_trace_insts.extend(disable_trace_at_start[:-1])
        instructions = disable_trace_at_start + instructions
    instructions.extend(final_insts)
    fix_instructions_for_assemble(instructions, code_options)
    # print("guarded code")
    # print(format_insts(instructions))
    # print("code_options")
    # for k, v in code_options.items():
    #     print(k, v)
    code_map = generate_code_map(original_instructions, instructions,
                                 in_trace_insts, next_original_pc)
    new_code = assemble_instructions(instructions, code_options)[1]
    return new_code, code_map


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
