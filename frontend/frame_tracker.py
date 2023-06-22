from typing import Callable, Any
from dataclasses import dataclass
from frontend.instruction import Instruction
from frontend.c_api import add_to_cache


@dataclass
class TracedCode:
    guard_fn: Callable[..., Any]
    graph_fn: Callable[..., Any]
    start_pc: int
    end_pc: int
    call_graph_insts: list[Instruction]


class FrameTracker:
    frame_id: int
    traced_codes: dict[int,
                       list[TracedCode]]  # start_pc -> list of traced codes
    callsite_id: dict[int, int]  # start_pc -> callsite_id

    def __init__(self, frame_id: int) -> None:
        self.frame_id = frame_id
        self.traced_codes = {0: []}
        self.callsite_id = {0: 0}

    def add(self, traced_code: TracedCode) -> None:
        start_pc = traced_code.start_pc
        assert traced_code.end_pc >= 0
        if start_pc not in self.traced_codes:
            self.traced_codes[start_pc] = []
            self.callsite_id[start_pc] = len(self.traced_codes) - 1

        self.traced_codes[start_pc].append(traced_code)

        add_to_cache(self.frame_id, self.callsite_id[start_pc],
                     len(self.traced_codes[start_pc]) - 1, traced_code.guard_fn,
                     traced_code.graph_fn)


trackers: dict[int, FrameTracker] = {}


def get_frame_tracker(frame_id: int) -> FrameTracker:
    return trackers[frame_id]


def enable_track(frame_id: int) -> None:
    if frame_id not in trackers:
        trackers[frame_id] = FrameTracker(frame_id)
