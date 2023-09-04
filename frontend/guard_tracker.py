from types import FrameType
from typing import Dict, Any
from abc import ABC, abstractmethod
import logging
import torch
from .code import ProcessedCode, load_code
from .c_api import get_value_stack_from_top, get_value_stack_size
from .instruction import Instruction, ci
from .cache import CachedGraph, get_frame_cache
from . import variables as vs
from .utils import is_scalar
from .object_table import ObjectTable
from .pycode_generator import GraphFnCodegen, GuardFnCodegen


class State:
    objects: ObjectTable
    start_pc: int
    start_stack_size: int
    is_empty: bool

    def __init__(self) -> None:
        self.objects = ObjectTable()
        self.start_pc = -1
        self.start_stack_size = -1
        self.is_empty = True

    def update(self, modifiers: list['StateModifier']) -> None:
        for modifier in modifiers:
            modifier.apply(self)
        self.is_empty = False

    @classmethod
    def from_frame(cls, frame: FrameType, read_stack: bool) -> 'State':
        state = cls()
        if read_stack:
            state.start_stack_size = get_value_stack_size(frame)
            for i in range(state.start_stack_size):
                value = get_value_stack_from_top(frame, i)
                var = vs.make_var_from_value(value, True,
                                             f"locals['__stack__{i}']")
                state.objects.add(var, value)
        return state


class StateModifier(ABC):

    @abstractmethod
    def apply(self, state: State) -> None:
        raise NotImplementedError()


class NewVarModifier(StateModifier):

    def __init__(self, var: vs.Variable, value: Any):
        self.var = var
        self.value = value

    def apply(self, state: State) -> None:
        state.objects.add(self.var, self.value)


class GuardTracker:
    code: ProcessedCode
    frame_id: int
    frame: FrameType
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
        self.have_error = False
        self.modifiers = []

    def record(
            self, frame: FrameType, frame_id: int
    ) -> None:  # pass frame and frame_id only for assertion
        assert frame_id == self.frame_id
        assert frame == self.frame

        inst = self.code.get_orig_inst(self.frame.f_lasti)
        if inst is None:
            self.restart(f"running injected code (pc={self.frame.f_lasti})")
            return
        # call init_state after is_inject_code check to avoid frequent init_state
        if self.have_error:
            self.init_state()
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
        assert self.state.start_pc >= 0
        end_pc = self.code.get_orig_pc(self.frame.f_lasti)
        if end_pc == -1:
            end_pc = self.code.get_next_orig_pc(self.frame.f_lasti)
        print("commiting", self.state.start_pc, end_pc)
        guard_codegen = GuardFnCodegen()
        for var in self.state.objects.get_all():
            var.make_guard(guard_codegen)
        guard_code = guard_codegen.get_code()
        print("guard_code:\n", guard_code)
        # TODO: can be optimized by only reproduce the modified variables
        graph_codegen = GraphFnCodegen()
        stack_size = get_value_stack_size(self.frame)
        for i in range(stack_size):
            value = get_value_stack_from_top(self.frame, i)
            var = vs.make_var_from_value(
                value, False)  # should we read from object table?
            var.make_output(f"__stack__{i}", graph_codegen)
        graph_code = graph_codegen.get_code()

        py_code = f"""\
{graph_code}
{guard_code}
        """
        out: Dict[str, Any] = dict()
        print("RUNNING PY CODE\n", py_code)
        exec(py_code, self.frame.f_globals, out)
        guard_fn = out["___make_guard_fn"]()
        graph_fn = out["___make_graph_fn"]()

        print("guard_fn:", guard_fn)
        print("pc:", self.state.start_pc, end_pc)
        print("stack:", self.state.start_stack_size, stack_size)

        get_frame_cache(self.frame_id).add(
            CachedGraph(
                guard_fn,
                graph_fn,
                self.state.start_pc,
                end_pc,
                start_stack_size=self.state.start_stack_size,
                end_stack_size=stack_size,
                return_values=graph_codegen.get_return_values(),
            ))
        self.state.is_empty = True

    def restart(self, restart_reason: str) -> None:
        logging.info(f"restart: {restart_reason}")
        self.have_error = True
        self.commit()

    def update_state(self) -> None:
        self.state.update(self.modifiers)
        self.modifiers.clear()

    def LOAD_FAST(self, inst: Instruction) -> None:
        obj = self.frame.f_locals[inst.argval]
        var = vs.make_var_from_value(obj, True, f'locals["{inst.argval}"]')
        self.modifiers.append(NewVarModifier(var, obj))

    def LOAD_CONST(self, _inst: Instruction) -> None:
        pass

    def BINARY_ADD(self, _inst: Instruction) -> None:
        obj1 = get_value_stack_from_top(self.frame, 1)
        obj2 = get_value_stack_from_top(self.frame, 0)
        if is_scalar(obj1) and is_scalar(obj2):
            pass
        else:
            # TODO: record the operation in compute graph
            raise NotImplementedError

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
