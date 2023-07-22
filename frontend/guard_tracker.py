from types import FrameType
from typing import Optional, Dict, Any
from frontend.frame_saver import ProcessedCode, load_frame
from frontend.c_api import get_value_stack_from_top, mark_need_postprocess
from dataclasses import dataclass
from frontend.instruction import Instruction, ci
from .frame_tracker import TracedCode, get_frame_tracker
import logging


@dataclass
class Guard:
    code: list[str]


@dataclass
class Variable:
    guard: Guard
    extract_code: str
    extract_insts: list[Instruction]


class CommitCtx:
    old_commiting: bool

    def __init__(self, tracker: 'GuardTracker'):
        self.tracker = tracker

    def __enter__(self) -> None:
        self.old_commiting = self.tracker.commiting
        self.tracker.commiting = True

    def __exit__(self, _exc_type: Any, _exc_value: Any,
                 _traceback: Any) -> None:
        self.tracker.commiting = self.old_commiting


class GuardTracker:
    stack: list[Optional[Variable]]
    code: ProcessedCode
    frame_id: int
    guard: Guard
    frame: FrameType
    is_commiting: bool
    error_in_commiting: bool
    start_pc: Optional[int]
    is_empty: bool

    def __init__(self, frame: FrameType, frame_id: int):
        self.code = load_frame(frame_id)
        self.frame = frame
        self.frame_id = frame_id
        self.init_state()

    def init_state(self) -> None:
        self.error = False
        self.stack = []
        self.guard = Guard([])
        self.error_in_commiting = False
        self.commiting = False
        self.start_pc = None
        self.is_empty = True

    def record(
            self, frame: FrameType, frame_id: int
    ) -> None:  # pass frame and frame_id only for assertion
        assert frame_id == self.frame_id
        assert frame == self.frame

        inst = self.code.get_orig_inst(self.frame.f_lasti)
        if inst is None:
            self.restart(f"running injected code (pc={self.frame.f_lasti})")
            return
        if self.start_pc is None:
            self.start_pc = self.code.get_orig_pc(self.frame.f_lasti)
            assert self.start_pc >= 0
        if hasattr(self, inst.opname):
            succuss = getattr(self, inst.opname)(inst)
            if succuss:
                self.is_empty = False
        else:
            self.restart(f"unknown opcode {inst.opname}")

    def commit(self) -> None:
        with CommitCtx(self):
            if self.is_empty:
                return
            end_pc = self.code.get_orig_pc(self.frame.f_lasti)
            if end_pc == -1:
                end_pc = self.code.get_next_orig_pc(self.frame.f_lasti)
            print("commiting", self.start_pc, end_pc)
            call_graph_insts = [
                ci("CALL_FUNCTION", 0),
                ci("POP_TOP"),
            ]
            for i, var in enumerate(self.stack):
                if var is not None:
                    self.guard.code.extend(var.guard.code)
                    call_graph_insts.extend(var.extract_insts)
                else:
                    val = get_value_stack_from_top(self.frame,
                                                   len(self.stack) - i - 1)
                    call_graph_insts.extend(self.create_var_insts(val))
            if self.error_in_commiting:
                return
            self.guarded_pop(len(self.stack))
            if self.error_in_commiting:
                return
            code = " and ".join(self.guard.code)
            if code == "":
                ok_code = "ok = True"
            else:
                ok_code = f"ok = {code}"
            py_code = f"""\
def ___make_guard_fn():
    def fn(locals):
        print("running guard_fn", locals)
        {ok_code}
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

            print("guard_fn:", guard_fn)
            print("call_graph_insts:", call_graph_insts)
            print("pc:", self.start_pc, end_pc)
            assert self.start_pc is not None
            get_frame_tracker(self.frame_id).add(
                TracedCode(
                    guard_fn,
                    graph_fn,
                    self.start_pc,
                    end_pc,
                    call_graph_insts,
                ))

    def create_var_insts(self, value: Any) -> list[Instruction]:
        if isinstance(value, int):
            return [ci("LOAD_CONST", value)]
        else:
            self.restart("unknown type in create_var_insts")
            return []

    def restart(self, restart_reason: str) -> None:
        if self.commiting:
            self.error_in_commiting = True
            return
        logging.info(f"restart: {restart_reason}")
        self.commit()
        self.init_state()

    def guarded_pop(self, num_var: int) -> None:
        for i in range(num_var):
            out = self.stack.pop()
            if out is None:
                continue
            value = get_value_stack_from_top(self.frame, i)
            if hasattr(self, f'add_guard_{type(value).__name__}'):
                getattr(self, f'add_guard_{type(value).__name__}')(out, value)
            else:
                self.restart("unknown type in add_guard")

    def add_guard_int(self, var: Variable, value: int) -> None:
        self.guard.code.extend(var.guard.code)
        self.guard.code.append(f"{var.extract_code} == {value}")

    def LOAD_FAST(self, inst: Instruction) -> bool:
        self.stack.append(
            Variable(Guard([]), f"locals['{inst.argval}']",
                     [ci('LOAD_FAST', inst.arg, inst.argval)]))
        return True

    def LOAD_CONST(self, inst: Instruction) -> bool:
        self.stack.append(None)
        return True

    def BINARY_ADD(self, inst: Instruction) -> bool:
        self.guarded_pop(2)
        self.stack.append(None)
        return True

    def RETURN_VALUE(self, inst: Instruction) -> bool:
        self.restart("return value")
        return False


trackers: list[GuardTracker] = []


def push_tracker(frame: FrameType, frame_id: int) -> None:
    print("init tracker", frame_id, "frame", hex(id(frame)), "frame_id",
          frame_id)
    trackers.append(GuardTracker(frame, frame_id))


def pop_tracker(frame_id: int) -> None:
    to_pop = trackers.pop()
    assert to_pop.is_empty
    assert to_pop.frame_id == frame_id


def record(frame: FrameType, frame_id: int) -> None:
    trackers[-1].record(frame, frame_id)


def reset() -> None:
    trackers.clear()
