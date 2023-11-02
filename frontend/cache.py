from types import CodeType
from typing import Callable, Any, Optional
from dataclasses import dataclass

from frontend.code import ProcessedCode
from .instruction import Instruction
from .c_api import add_to_cache
from .store_pos import StorePos


@dataclass
class CachedGraph:
    guard_fn: Callable[..., Any]
    graph_fn: Callable[..., Any]
    start_pc: int
    end_pc: int
    start_stack_size: int
    end_stack_size: int
    return_values: list[StorePos]
    key: int
    object_refs: list[Any]


TOTAL_SIZE = 0


class FrameCache:
    frame_id: int
    cached_graphs: dict[int,
                        list[CachedGraph]]  # start_pc -> list of cached graph
    callsite_id: dict[int, int]  # start_pc -> callsite_id
    pre_cache_size: int
    new_code: Optional[CodeType]
    code_map: Optional[ProcessedCode]
    updated: bool

    def __init__(self, frame_id: int) -> None:
        self.frame_id = frame_id
        self.cached_graphs = {0: []}
        self.callsite_id = {0: 0}
        self.new_code = None
        self.code_map = None
        self.updated = True  # rewrite bytecode for the first time

    def add(self, traced_code: CachedGraph) -> None:
        start_pc = traced_code.start_pc
        assert traced_code.end_pc >= 0
        if start_pc not in self.cached_graphs:
            self.cached_graphs[start_pc] = []
            self.callsite_id[start_pc] = len(self.cached_graphs) - 1

        self.cached_graphs[start_pc].append(traced_code)

        add_to_cache(self.frame_id, self.callsite_id[start_pc],
                     len(self.cached_graphs[start_pc]) - 1,
                     traced_code.guard_fn, traced_code.graph_fn)
        global TOTAL_SIZE
        TOTAL_SIZE += 1
        self.updated = True

    def set_new_code(self, new_code: CodeType, code_map: ProcessedCode) -> None:
        self.new_code = new_code
        self.code_map = code_map
        self.updated = False


frame_caches: dict[int, FrameCache] = {}


def get_frame_cache(frame_id: int) -> FrameCache:
    return frame_caches[frame_id]


def enable_cache(frame_id: int) -> None:
    if frame_id not in frame_caches:
        frame_caches[frame_id] = FrameCache(frame_id)


def check_cache_updated(frame_id: int) -> bool:
    assert frame_id in frame_caches
    return frame_caches[frame_id].updated


def reset() -> None:
    global TOTAL_SIZE
    TOTAL_SIZE = 0
    frame_caches.clear()
