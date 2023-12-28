import dataclasses
import dis
import sys
import functools
from typing import Union, List, Union
from collections import deque
from .instruction import Instruction

TERMINAL_OPCODES = {
    dis.opmap["RETURN_VALUE"],
    dis.opmap["JUMP_FORWARD"],
    dis.opmap["RAISE_VARARGS"],
}
if sys.version_info >= (3, 9):
    TERMINAL_OPCODES.add(dis.opmap["RERAISE"])
if sys.version_info >= (3, 11):
    TERMINAL_OPCODES.add(dis.opmap["JUMP_BACKWARD"])
else:
    TERMINAL_OPCODES.add(dis.opmap["JUMP_ABSOLUTE"])
JUMP_OPCODES = set(dis.hasjrel + dis.hasjabs)
JUMP_OPNAMES = {dis.opname[opcode] for opcode in JUMP_OPCODES}
MUST_JUMP_OPCODES = {
    dis.opmap["JUMP_FORWARD"],
    dis.opmap["JUMP_ABSOLUTE"],
}
HASLOCAL = set(dis.haslocal)
HASFREE = set(dis.hasfree)
jump_only_opnames = ['JUMP_FORWARD', 'JUMP_ABSOLUTE']
jump_or_next_opnames = [
    'POP_JUMP_IF_TRUE', 'POP_JUMP_IF_FALSE', 'JUMP_IF_NOT_EXC_MATCH',
    'JUMP_IF_TRUE_OR_POP', 'JUMP_IF_FALSE_OR_POP', 'FOR_ITER'
]
jump_only_opcodes = [dis.opmap[opname] for opname in jump_only_opnames]
jump_or_next_opcodes = [dis.opmap[opname] for opname in jump_or_next_opnames]


def get_indexof(insts: List[Instruction]) -> dict[Instruction, int]:
    """
    Get a mapping from instruction memory address to index in instruction list.
    Additionally checks that each instruction only appears once in the list.
    """
    indexof: dict[Instruction, int] = {}
    for i, inst in enumerate(insts):
        assert inst not in indexof
        indexof[inst] = i
    return indexof


@dataclasses.dataclass
class ReadsWrites:
    reads: set[str]
    writes: set[str]
    visited: set[int]


def livevars_analysis(instructions: List[Instruction],
                      instruction: Instruction) -> set[str]:
    indexof = get_indexof(instructions)

    prev: dict[int, list[int]] = {}
    succ: dict[int, list[int]] = {}
    prev[0] = []
    for i, inst in enumerate(instructions):
        if inst.opcode not in TERMINAL_OPCODES:
            prev[i + 1] = [i]
            succ[i] = [i + 1]
        else:
            prev[i + 1] = []
            succ[i] = []
    for i, inst in enumerate(instructions):
        if inst.opcode in JUMP_OPCODES:
            assert inst.target is not None
            target_pc = indexof[inst.target]
            prev[target_pc].append(i)
            succ[i].append(target_pc)

    live_vars: dict[int, frozenset[str]] = {}

    start_pc = indexof[instruction]
    to_visit = deque([
        pc for pc in range(len(instructions))
        if instructions[pc].opcode in TERMINAL_OPCODES
    ])
    in_progress: set[int] = set(to_visit)

    def join_fn(a: frozenset[str], b: frozenset[str]) -> frozenset[str]:
        return frozenset(a | b)

    def gen_fn(
            inst: Instruction,
            incoming: frozenset[str]) -> tuple[frozenset[str], frozenset[str]]:
        gen = set()
        kill = set()
        if inst.opcode in HASLOCAL or inst.opcode in HASFREE:
            if "LOAD" in inst.opname or "DELETE" in inst.opname:
                assert isinstance(inst.argval, str)
                gen.add(inst.argval)
            elif "STORE" in inst.opname:
                assert isinstance(inst.argval, str)
                kill.add(inst.argval)
            elif inst.opname == "MAKE_CELL":
                pass
            else:
                raise NotImplementedError(f"unhandled {inst.opname}")

        return frozenset(gen), frozenset(kill)

    while len(to_visit) > 0:
        pc = to_visit.popleft()
        in_progress.remove(pc)
        if pc in live_vars:
            before = hash(live_vars[pc])
        else:
            before = None
        succs = [
            live_vars[succ_pc] for succ_pc in succ[pc] if succ_pc in live_vars
        ]
        if len(succs) > 0:
            incoming = functools.reduce(join_fn, succs)
        else:
            incoming = frozenset()

        gen, kill = gen_fn(instructions[pc], incoming)

        out = (incoming - kill) | gen
        live_vars[pc] = out
        if hash(out) != before:
            for prev_pc in prev[pc]:
                if prev_pc not in in_progress:
                    to_visit.append(prev_pc)
                    in_progress.add(prev_pc)
    return set(live_vars[start_pc])


stack_effect = dis.stack_effect


@dataclasses.dataclass
class FixedPointBox:
    value: bool = True


Inf = float  # assume to be float("inf") or float("-inf")


@dataclasses.dataclass
class StackSize:
    low: Union[int, Inf]
    high: Union[int, Inf]
    fixed_point: FixedPointBox

    def zero(self) -> None:
        self.low = 0
        self.high = 0
        self.fixed_point.value = False

    def offset_of(self, other: 'StackSize', n: int) -> None:
        prior = (self.low, self.high)
        self.low = min(self.low, other.low + n)
        self.high = max(self.high, other.high + n)
        if (self.low, self.high) != prior:
            self.fixed_point.value = False


def stacksize_analysis(instructions: List[Instruction]) -> int:
    assert instructions
    fixed_point = FixedPointBox()
    stack_sizes = {
        inst: StackSize(float("inf"), float("-inf"), fixed_point)
        for inst in instructions
    }
    stack_sizes[instructions[0]].zero()

    for _ in range(100):
        if fixed_point.value:
            break
        fixed_point.value = True

        for inst, next_inst in zip(instructions, instructions[1:] + [None]):
            stack_size = stack_sizes[inst]
            if inst.opcode not in TERMINAL_OPCODES:
                assert next_inst is not None, f"missing next inst: {inst}"
                stack_sizes[next_inst].offset_of(
                    stack_size, stack_effect(inst.opcode, inst.arg, jump=False))
            if inst.opcode in JUMP_OPCODES:
                assert inst.target is not None, f"missing target: {inst}"
                stack_sizes[inst.target].offset_of(
                    stack_size, stack_effect(inst.opcode, inst.arg, jump=True))

    if False:
        for i, inst in enumerate(instructions):
            stack_size = stack_sizes[inst]
            print(i, ":", stack_size.low, stack_size.high, inst)

    low = min([x.low for x in stack_sizes.values()])
    high = max([x.high for x in stack_sizes.values()])

    assert fixed_point.value, "failed to reach fixed point"
    assert low >= 0
    assert isinstance(high, int)  # not infinity
    return high


def end_of_control_flow(instructions: List[Instruction], start_pc: int) -> int:
    """
    Find the end of the control flow block starting at the given instruction.
    """
    while instructions[start_pc].opname == 'EXTENDED_ARG':
        start_pc += 1
    assert instructions[start_pc].opcode in JUMP_OPCODES
    assert instructions[start_pc].target is not None
    indexof = get_indexof(instructions)
    return_value_opcode = dis.opmap['RETURN_VALUE']
    possible_end_pcs = set()
    for end_pc, inst in enumerate(instructions):
        if end_pc == start_pc:
            continue
        inst = instructions[end_pc]
        if not inst.is_jump_target:
            continue
        visited = set()
        queue = deque([start_pc])
        reach_end = False
        while queue and not reach_end:
            pc = queue.popleft()
            inst = instructions[pc]
            targets: list[int] = []
            if inst.target is not None:
                if inst.opcode in jump_only_opcodes:
                    targets = [indexof[inst.target]]
                elif inst.opcode in jump_or_next_opcodes:
                    targets = [indexof[inst.target], pc + 1]
                else:
                    raise NotImplementedError(f"unhandled {inst.opname}")
            else:
                targets = [pc + 1]
            for target in targets:
                if instructions[target].opcode == return_value_opcode:
                    reach_end = True
                    break
                if target in visited:
                    continue
                if target == end_pc:
                    continue
                visited.add(target)
                queue.append(target)
        if not reach_end:
            possible_end_pcs.add(end_pc)
    visited = set()
    dist: dict[int, int] = {start_pc: 0}
    queue = deque([start_pc])
    while queue:
        pc = queue.popleft()
        inst = instructions[pc]
        if inst.opcode == return_value_opcode:
            continue
        targets = []
        if inst.target is not None:
            if inst.opcode in jump_only_opcodes:
                targets = [indexof[inst.target]]
            elif inst.opcode in jump_or_next_opcodes:
                targets = [indexof[inst.target], pc + 1]
            else:
                raise NotImplementedError(f"unhandled {inst.opname}")
        else:
            targets = [pc + 1]
        for target in targets:
            if target in visited:
                continue
            visited.add(target)
            dist[target] = dist[pc] + 1
            queue.append(target)
    min_dist = min([dist[end_pc] for end_pc in possible_end_pcs])
    for end_pc in possible_end_pcs:
        if dist[end_pc] == min_dist:
            return end_pc
    return -1


def eliminate_dead_code(
        instructions: list[Instruction],
        start_pc: Union[Instruction, int] = -1,
        end_pcs: list[Union[Instruction, int]] = []) -> list[int]:
    """
    Eliminate dead code in the given instruction list.
    """
    if start_pc == -1:
        start_pc = 0
    if len(end_pcs) == 0:
        return_value_opcode = dis.opmap['RETURN_VALUE']
        end_pcs = [
            i for i, inst in enumerate(instructions)
            if inst.opcode == return_value_opcode
        ]
    indexof = get_indexof(instructions)
    for i, inst_or_pc in enumerate(end_pcs):
        if isinstance(inst_or_pc, Instruction):
            end_pcs[i] = indexof[inst_or_pc]
    if isinstance(start_pc, Instruction):
        start_pc = indexof[start_pc]
    visited = set()
    queue = deque([start_pc])
    visited.add(start_pc)
    while queue:
        pc = queue.popleft()
        inst = instructions[pc]
        if pc in end_pcs:
            continue
        targets = []
        if inst.target is not None:
            if inst.opcode in jump_only_opcodes:
                targets = [indexof[inst.target]]
            elif inst.opcode in jump_or_next_opcodes:
                targets = [indexof[inst.target], pc + 1]
            else:
                raise NotImplementedError(f"unhandled {inst.opname}")
        else:
            targets = [pc + 1]
        for target in targets:
            if target in visited:
                continue
            visited.add(target)
            queue.append(target)
    visited_list = list(visited)
    visited_list.sort()
    return visited_list
