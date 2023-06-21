from types import FrameType
from typing import Optional
from frontend.frame_saver import ProcessedCode, load_frame
from frontend.c_api import get_value_stack_from_top
from dataclasses import dataclass
from frontend.instruction import Instruction


@dataclass
class Guard:
    code: list[str]


@dataclass
class Variable:
    guard: Guard
    extract_code: str


class GuardTracker:
    stack: list[Optional[Variable]]
    code: ProcessedCode
    frame_id: int
    guard: Guard
    frame: Optional[FrameType]

    def __init__(self, frame_id: int = 0):
        self.stack = []
        self.code = load_frame(frame_id)
        self.frame_id = frame_id
        self.guard = Guard([])
        self.frame = None

    def record(self, frame: FrameType, frame_id: int) -> None:
        assert frame_id == self.frame_id
        self.frame = frame

        inst = self.code.get_orig_inst(frame.f_lasti)
        if inst is None:
            self.frame = None
            return
        print("extracted", inst.opname)
        if hasattr(self, inst.opname):
            getattr(self, inst.opname)(inst)
        else:
            self.restart()

    def restart(self) -> None:
        raise NotImplementedError

    def guarded_pop(self, num_var: int) -> None:
        for i in range(num_var):
            out = self.stack.pop()
            if out is None:
                continue
            assert self.frame is not None
            value = get_value_stack_from_top(self.frame, i)
            print("value:", value)
            if hasattr(self, f'add_guard_{type(value).__name__}'):
                getattr(self, f'add_guard_{type(value).__name__}')(out, value)
            else:
                self.restart()

    def add_guard_int(self, var: Variable, value: int) -> None:
        self.guard.code.extend(var.guard.code)
        self.guard.code.append(f"{var.extract_code} == {value}")

    def LOAD_FAST(self, inst: Instruction) -> None:
        self.stack.append(Variable(Guard([]), f"locals['{inst.argval}']"))

    def LOAD_CONST(self, inst: Instruction) -> None:
        self.stack.append(None)

    def BINARY_ADD(self, inst: Instruction) -> None:
        self.guarded_pop(2)
        self.stack.append(None)


trackers: list[GuardTracker] = []


def push_tracker(frame_id: int) -> None:
    trackers.append(GuardTracker(frame_id))


def pop_tracker(frame_id: int) -> None:
    to_pop = trackers.pop()
    print("guard", to_pop.guard.code)
    assert to_pop.frame_id == frame_id


def record(frame: FrameType, frame_id: int) -> None:
    trackers[-1].record(frame, frame_id)
