from frontend.instruction import Instruction
from typing import Optional, cast
import dis
import logging

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

    last_pc_guarded_to_origin: dict[int, int]  # last pc guard -> origin
    # heheda: not sure whether we need this field
    next_pc_guarded_to_origin: dict[
        int, int]  # next pc guard -> origin, -1 if unknown
    original_insts: list[Instruction]
    guard_insts: list[Instruction]
    original_pc: dict[Instruction,
                      int]  # original instruction -> pc in original_insts
    guarded_pc: dict[Instruction,
                     int]  # guarded instruction -> pc in guard_insts

    def __init__(self, original_insts: list[Instruction],
                 guard_insts: list[Instruction],
                 inside_trace_opcodes: list[Instruction]) -> None:
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

        self.last_pc_guarded_to_origin = {}
        for inst in guard_insts:
            if inst.original_inst is not None:
                self.last_pc_guarded_to_origin[cast(int, inst.offset) //
                                               2] = self.original_pc[
                                                   inst.original_inst]
        for inst in inside_trace_opcodes:
            self.last_pc_guarded_to_origin[cast(int, inst.offset) // 2] = -1

        self.next_pc_guarded_to_origin = {}
        for i, inst in enumerate(guard_insts):
            if inst.original_inst is not None:
                if inst.opname in dynamic_next_pc_opnames:
                    self.next_pc_guarded_to_origin[cast(int, inst.offset) //
                                                   2] = -1
                elif inst.opcode in dis.hasjabs or inst.opcode in dis.hasjrel:
                    if inst.opname in ('JUMP_ABSOLUTE', 'JUMP_FORWARD'):
                        assert inst.target in self.guarded_pc
                        guarded_pc = self.guarded_pc[inst.target]
                        self.next_pc_guarded_to_origin[cast(int, inst.offset) // 2] \
                            = self.last_pc_guarded_to_origin[guarded_pc]
                    else:
                        logging.info("unknown jump inst %s", inst)
                else:
                    self.next_pc_guarded_to_origin[cast(
                        int, inst.offset) // 2] = self.get_pc(
                            self.original_insts,
                            self.original_pc[inst.original_inst] + 1)

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
        if pc not in self.last_pc_guarded_to_origin:
            return -2

        return self.last_pc_guarded_to_origin[pc]

    def get_orig_inst(self, lasti: int) -> Optional[Instruction]:
        pc = lasti // 2
        assert pc in self.last_pc_guarded_to_origin
        origin_pc = self.last_pc_guarded_to_origin[pc]
        if origin_pc == -1:
            return None  # is a helper opcode inside tracing region
        return self.original_insts[self.last_pc_guarded_to_origin[pc]]

    def get_next_orig_pc(self,
                         lasti: int,
                         last_tos_bool: Optional[bool] = None) -> int:
        pc = lasti // 2
        if pc not in self.next_pc_guarded_to_origin:
            return -1
        next_pc = self.next_pc_guarded_to_origin[pc]
        if next_pc != -1:
            return next_pc
        last_origin_pc = self.last_pc_guarded_to_origin[pc]
        last_inst = self.original_insts[last_origin_pc]
        if last_inst.opname in ("POP_JUMP_IF_FALSE", "JUMP_IF_FALSE_OR_POP"):
            if last_tos_bool is None:
                raise ValueError(
                    "last_tos_bool is None when last_inst is a conditional jump"
                )
            if last_tos_bool:
                return self.get_pc(self.original_insts, last_origin_pc + 1)
            else:
                assert last_inst.target is not None
                return self.original_pc[last_inst.target]
        elif last_inst.opname in ("POP_JUMP_IF_TRUE", "JUMP_IF_TRUE_OR_POP"):
            if last_tos_bool is None:
                raise ValueError(
                    "last_tos_bool is None when last_inst is a conditional jump"
                )
            if last_tos_bool:
                assert last_inst.target is not None
                return self.original_pc[last_inst.target]
            else:
                return self.get_pc(self.original_insts, last_origin_pc + 1)

        else:
            raise NotImplementedError

    def get_dependence_of_stack_var(self, original_inst: Instruction,
                                    stack_depth: int) -> list[Instruction]:
        raise NotImplementedError

    def get_dependence_of_local_var(self, original_inst: Instruction,
                                    local_name: str) -> list[Instruction]:
        raise NotImplementedError


processed_codes: dict[int, ProcessedCode] = {}  # frame_id -> ProcessedCode


def save_frame(original_insts: list[Instruction],
               generated_insts: list[Instruction], frame_id: int,
               inside_trace_opcodes: list[Instruction]) -> None:
    processed_codes[frame_id] = ProcessedCode(original_insts, generated_insts,
                                              inside_trace_opcodes)


def load_frame(frame_id: int) -> ProcessedCode:
    return processed_codes[frame_id]