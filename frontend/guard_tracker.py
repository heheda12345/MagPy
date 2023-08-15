from types import FrameType
from typing import Dict, Any
from abc import ABC, abstractmethod
import logging
import re
import torch
from .code import ProcessedCode, load_code
from .c_api import get_value_stack_from_top, get_value_stack_size
from .instruction import Instruction, ci
from .cache import CachedGraph, get_frame_cache
from . import variables as var


class State:
    guard: var.Guard
    start_pc: int
    is_empty: bool
    stack: list[var.Variable]

    def __init__(self) -> None:
        self.guard = var.Guard([])
        self.start_pc = -1
        self.is_empty = True
        self.stack = []

    def update(self, modifiers: list['StateModifier']) -> None:
        for modifier in modifiers:
            modifier.apply(self)
        self.is_empty = False

    @classmethod
    def from_frame(cls, frame: FrameType, read_stack: bool) -> 'State':
        state = cls()
        if read_stack:
            stack_size = get_value_stack_size(frame)
            state.stack = [var.StackVar(i) for i in range(stack_size)]
        return state


class StateModifier(ABC):

    @abstractmethod
    def apply(self, state: State) -> None:
        raise NotImplementedError()


class NewGuardModifier(StateModifier):

    def __init__(self, guard: var.Guard):
        self.guard = guard

    def apply(self, state: State) -> None:
        state.guard.code.extend(self.guard.code)


class StackPopModifier(StateModifier):
    n_pop: int

    def __init__(self, n_pop: int):
        self.n_pop = n_pop

    def apply(self, state: State) -> None:
        for _ in range(self.n_pop):
            state.stack.pop()


class StackPushModifier(StateModifier):
    var_push: var.Variable

    def __init__(self, var_push: var.Variable):
        self.var_push = var_push

    def apply(self, state: State) -> None:
        state.stack.append(self.var_push)


class GuardTracker:
    code: ProcessedCode
    frame_id: int
    frame: FrameType
    is_commiting: bool
    state: State
    modifiers: list[StateModifier]
    have_error: bool

    def __init__(self, frame: FrameType, frame_id: int):
        self.code = load_code(frame_id)
        self.frame = frame
        self.frame_id = frame_id
        self.init_state(
            read_stack=False
        )  # stack pointer is not initialized at the creation of a stack frame

    def init_state(self, read_stack: bool = True) -> None:
        self.state = State.from_frame(self.frame, read_stack)
        self.is_commiting = False
        self.have_error = False
        self.error_in_commiting = False
        self.modifiers = []

    def record(
            self, frame: FrameType, frame_id: int
    ) -> None:  # pass frame and frame_id only for assertion
        assert frame_id == self.frame_id
        assert frame == self.frame

        if self.have_error:
            self.init_state()

        inst = self.code.get_orig_inst(self.frame.f_lasti)
        if inst is None:
            self.restart(f"running injected code (pc={self.frame.f_lasti})")
            return
        if self.state.start_pc == -1:
            self.state.start_pc = self.code.get_orig_pc(self.frame.f_lasti)
            assert self.state.start_pc >= 0
        if hasattr(self, inst.opname):
            getattr(self, inst.opname)(inst)
            if not self.have_error:
                self.update_state()
        else:
            self.restart(f"unknown opcode {inst.opname}")

    def commit(self) -> None:
        if self.state.is_empty:
            return
        end_pc = self.code.get_orig_pc(self.frame.f_lasti)
        if end_pc == -1:
            end_pc = self.code.get_next_orig_pc(self.frame.f_lasti)
        assert self.is_commiting == False
        self.is_commiting = True
        print("commiting", self.state.start_pc, end_pc)
        call_graph_insts = [
            ci("CALL_FUNCTION", 0),
            ci("POP_TOP"),
        ]
        guard = self.state.guard
        # FIXME: should not reproduce the whole stack
        for i, v in enumerate(self.state.stack):
            if not isinstance(v, var.RuntimeVar):
                guard.code.extend(v.guard.code)
                call_graph_insts.extend(v.extract_insts)
            else:
                val = get_value_stack_from_top(self.frame,
                                               len(self.state.stack) - i - 1)
                call_graph_insts.extend(self.create_var_insts(val))
        if self.error_in_commiting:
            return
        code = " and ".join(guard.code)
        if code == "":
            ok_code = "ok = True"
        else:
            ok_code = f"ok = {code}"
        py_code = f"""\
def ___make_guard_fn():
    def fn(locals):
        print("running guard_fn", locals)
        {ok_code}
        print("ok = ", ok)
        return ok
    return fn
def ___make_graph_fn():
    def fn():
        print("running graph_fn")
        return None
    return fn
            """
        out: Dict[str, Any] = dict()
        print("RUNNING PY CODE", py_code)
        exec(py_code, self.frame.f_globals, out)
        guard_fn = out["___make_guard_fn"]()
        graph_fn = out["___make_graph_fn"]()

        stack_var_max_depth = -1
        # find all __stack__* variables from py_code
        pattern = re.compile(r"__stack__(\d+)")
        for m in pattern.finditer(py_code):
            stack_var_max_depth = max(stack_var_max_depth, int(m.group(1)))
        stack_var_max_depth += 1

        print("guard_fn:", guard_fn)
        print("call_graph_insts:", call_graph_insts)
        print("pc:", self.state.start_pc, end_pc)
        assert self.state.start_pc >= 0
        get_frame_cache(self.frame_id).add(
            CachedGraph(
                guard_fn,
                graph_fn,
                self.state.start_pc,
                end_pc,
                call_graph_insts,
                stack_var_max_depth,
            ))
        self.is_commiting = False

    def create_var_insts(self, value: Any) -> list[Instruction]:
        if isinstance(value, int):
            return [ci("LOAD_CONST", value)]
        elif isinstance(value, torch.Tensor):
            return [ci("LOAD_CONST", 233)]
        else:
            self.restart("unknown type in create_var_insts")
            return []

    def restart(self, restart_reason: str) -> None:
        logging.info(
            f"restart (commiting = {self.is_commiting}): {restart_reason}")
        if self.is_commiting:
            self.error_in_commiting = True
            return
        self.have_error = True
        self.commit()

    def update_state(self) -> None:
        self.state.update(self.modifiers)
        self.modifiers.clear()

    def guarded_pop(self, num_var: int) -> None:
        for i in range(num_var):
            out = self.state.stack[-i - 1]
            if isinstance(out, var.RuntimeVar):
                continue
            value = get_value_stack_from_top(self.frame, i)
            type_name = type(value).__name__
            if hasattr(self, f'add_guard_{type_name}'):
                getattr(self, f'add_guard_{type_name}')(out, value)
            else:
                self.restart(f"unknown type {type_name} in add_guard")
        self.modifiers.append(StackPopModifier(num_var))

    def add_guard_int(self, var: var.Variable, value: int) -> None:
        self.state.guard.code.extend(var.guard.code)
        self.state.guard.code.append(f"{var.extract_code} == {value}")

    def add_guard_Tensor(self, var: var.Variable, value: torch.Tensor) -> None:
        self.state.guard.code.extend(var.guard.code)
        self.state.guard.code.append(
            f"{var.extract_code}.shape == {value.shape}")

    def LOAD_FAST(self, inst: Instruction) -> None:
        new_var = var.Variable(var.Guard([]), f"locals['{inst.argval}']",
                               [ci('LOAD_FAST', inst.arg, inst.argval)])
        self.modifiers.append(StackPushModifier(new_var))

    def LOAD_CONST(self, _inst: Instruction) -> None:
        self.modifiers.append(StackPushModifier(var.RuntimeVar()))

    def BINARY_ADD(self, _inst: Instruction) -> None:
        self.guarded_pop(2)
        self.modifiers.append(StackPushModifier(var.RuntimeVar()))

    def RETURN_VALUE(self, _inst: Instruction) -> None:
        self.restart("return value")


trackers: list[GuardTracker] = []


def push_tracker(frame: FrameType, frame_id: int) -> None:
    print("init tracker", frame_id, "frame", hex(id(frame)), "frame_id",
          frame_id)
    trackers.append(GuardTracker(frame, frame_id))


def pop_tracker(frame_id: int) -> None:
    to_pop = trackers.pop()
    assert to_pop.state.is_empty
    assert to_pop.frame_id == frame_id


def record(frame: FrameType, frame_id: int) -> None:
    trackers[-1].record(frame, frame_id)


def reset() -> None:
    trackers.clear()
