from typing import Optional, cast
import dis
from .instruction import Instruction

dynamic_next_pc_opnames = {
    "POP_JUMP_IF_FALSE",
    "POP_JUMP_IF_TRUE",
    "JUMP_IF_FALSE_OR_POP",
    "JUMP_IF_TRUE_OR_POP",
    "JUMP_IF_NOT_EXC_MATCH",
    "FOR_ITER",
}

dynamic_next_pc_opcodes = {
    dis.opmap[opname] for opname in dynamic_next_pc_opnames
}


class ProcessedCode:
    '''
    EXTENDED_ARG: last_pc and next_pc will both return the real instruction
    after EXTENDED_ARG
    Example:
    0 LOAD_FAST
    2 EXTENDED_ARG
    4 EXTENDED_ARG
    6 LOAD_CONST
    8 RETURN_VALUE
    last_i = 0:
        last_pc = original_insts[0] (LOAD_FAST)
        next_pc = original_insts[3] (LOAD_CONST)
    last_i = 2
        last_pc = original_insts[3] (LOAD_CONST)
        next_pc = original_insts[4] (RETURN_VALUE)
    last_i cannot be 4 or 6
    JUMP_IF_{TRUE,FALSE} like opcodes needs the current TOS to decide the next pc
    FOR_ITER: a NOP is inserted after FOR_ITER, the next pc of FOR_ITER is -1
    RETURN_VALUE: the next pc of RETURN_VALUE is len(original_insts)
    naming:
        last_i: the index before diviing by sizeof(Instruction)
        pc: the index after dividing by sizeof(Instruction)
    '''

    pc_guarded_to_origin: dict[int, int]  # last pc guard -> origin
    # heheda: not sure whether we need this field
    original_insts: list[Instruction]
    guard_insts: list[Instruction]
    original_pc: dict[Instruction,
                      int]  # original instruction -> pc in original_insts
    guarded_pc: dict[Instruction,
                     int]  # guarded instruction -> pc in guard_insts
    next_original_pc: dict[
        int,
        int]  # pc guarded -> original, only for replaced code in the orignal section of the guarded code

    def __init__(
            self, original_insts: list[Instruction],
            guard_insts: list[Instruction],
            inside_trace_opcodes: list[Instruction],
            next_original_pc: list[tuple[Instruction, Instruction]]) -> None:
        self.original_insts = original_insts[:]
        self.guard_insts = guard_insts[:]

        self.original_pc = {}
        pc = -1
        for inst in original_insts:
            assert inst.offset is not None
        for inst in guard_insts:
            assert inst.offset is not None
        for inst in reversed(original_insts):
            if inst.opname != "EXTENDED_ARG":
                pc = cast(int, inst.offset) // 2  # mypy: no-strict-optional
            self.original_pc[inst] = pc

        self.guarded_pc = {}
        pc = -1
        for inst in reversed(guard_insts):
            if inst.opname != "EXTENDED_ARG":
                pc = cast(int, inst.offset) // 2
            self.guarded_pc[inst] = pc

        self.pc_guarded_to_origin = {}
        for inst in guard_insts:
            if inst.original_inst is not None:
                self.pc_guarded_to_origin[cast(int, inst.offset) //
                                          2] = self.original_pc[
                                              inst.original_inst]
        for inst in inside_trace_opcodes:
            self.pc_guarded_to_origin[cast(int, inst.offset) // 2] = -1

        self.next_original_pc = {}
        for o, g in next_original_pc:
            self.next_original_pc[self.guarded_pc[g]] = self.original_pc[o]

    def get_pc(self, inst_list: list[Instruction], pc: int) -> int:
        while pc < len(inst_list) and inst_list[pc].opname == "EXTENDED_ARG":
            pc += 1
        return pc

    def get_orig_pc(self, lasti: int) -> int:
        '''
        returns -1 if the lasti is a helper opcode inside tracing region
        returns -2 if the lasti is outside tracing region
        '''
        pc = lasti // 2
        if pc not in self.pc_guarded_to_origin:
            return -2

        return self.pc_guarded_to_origin[pc]

    def get_orig_inst(self, lasti: int) -> Optional[Instruction]:
        pc = lasti // 2
        assert pc in self.pc_guarded_to_origin, (
            "pc %d not in pc_guarded_to_origin" % pc)
        origin_pc = self.pc_guarded_to_origin[pc]
        if origin_pc == -1:
            return None  # is a helper opcode inside tracing region
        return self.original_insts[self.pc_guarded_to_origin[pc]]

    def get_next_orig_pc(self, lasti: int) -> int:
        pc = lasti // 2
        if pc not in self.next_original_pc:
            raise ValueError("pc %d not in next_original_pc" % pc)

        return self.next_original_pc[pc]

    def get_inst(self, lasti: int) -> Instruction:
        pc = lasti // 2
        return self.guard_insts[pc]

    def get_dependence_of_stack_var(self, original_inst: Instruction,
                                    stack_depth: int) -> list[Instruction]:
        raise NotImplementedError

    def get_dependence_of_local_var(self, original_inst: Instruction,
                                    local_name: str) -> list[Instruction]:
        raise NotImplementedError


processed_codes: dict[int, ProcessedCode] = {}  # frame_id -> ProcessedCode


def save_code(original_insts: list[Instruction],
              generated_insts: list[Instruction], frame_id: int,
              inside_trace_opcodes: list[Instruction],
              next_original_pc: list[tuple[Instruction, Instruction]]) -> None:
    processed_codes[frame_id] = ProcessedCode(original_insts, generated_insts,
                                              inside_trace_opcodes,
                                              next_original_pc)


def load_code(frame_id: int) -> ProcessedCode:
    return processed_codes[frame_id]


def reset() -> None:
    global processed_codes
    processed_codes.clear()