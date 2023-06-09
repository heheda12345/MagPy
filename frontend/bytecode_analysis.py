import dataclasses
import dis
import sys
from numbers import Real

TERMINAL_OPCODES = {
    dis.opmap["RETURN_VALUE"],
    dis.opmap["JUMP_FORWARD"],
    dis.opmap["RAISE_VARARGS"],
    # TODO(jansel): double check exception handling
}
if sys.version_info >= (3, 9):
    TERMINAL_OPCODES.add(dis.opmap["RERAISE"])
if sys.version_info >= (3, 11):
    TERMINAL_OPCODES.add(dis.opmap["JUMP_BACKWARD"])
else:
    TERMINAL_OPCODES.add(dis.opmap["JUMP_ABSOLUTE"])
JUMP_OPCODES = set(dis.hasjrel + dis.hasjabs)
JUMP_OPNAMES = {dis.opname[opcode] for opcode in JUMP_OPCODES}
HASLOCAL = set(dis.haslocal)
HASFREE = set(dis.hasfree)

stack_effect = dis.stack_effect


@dataclasses.dataclass
class FixedPointBox:
    value: bool = True


@dataclasses.dataclass
class StackSize:
    low: Real
    high: Real
    fixed_point: FixedPointBox

    def zero(self):
        self.low = 0
        self.high = 0
        self.fixed_point.value = False

    def offset_of(self, other, n):
        prior = (self.low, self.high)
        self.low = min(self.low, other.low + n)
        self.high = max(self.high, other.high + n)
        if (self.low, self.high) != prior:
            self.fixed_point.value = False


def stacksize_analysis(instructions):
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
                stack_sizes[inst.target].offset_of(
                    stack_size, stack_effect(inst.opcode, inst.arg, jump=True))

    if False:
        for inst in instructions:
            stack_size = stack_sizes[inst]
            print(stack_size.low, stack_size.high, inst)

    low = min([x.low for x in stack_sizes.values()])
    high = max([x.high for x in stack_sizes.values()])

    assert fixed_point.value, "failed to reach fixed point"
    assert low >= 0
    return high
