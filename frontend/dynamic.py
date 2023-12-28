import dataclasses
from typing import Any


class Dynamic:
    pass


class ScalarWithUnknownValue(Dynamic):
    pass


@dataclasses.dataclass
class DynamicControlFlow(Dynamic):
    pc: int
    opcode: str


dynamic_vars = {}
dynamic_refs = {}
dynamic_pcs = {}
dynamic_need_branch_rewrite: dict[int, list[int]] = {}


def mark_dynamic(obj: Any, dyn: Dynamic) -> None:
    idx = id(obj)
    dynamic_vars[idx] = dyn
    dynamic_refs[idx] = obj


def contains(obj: Any) -> bool:
    idx = id(obj)
    return idx in dynamic_vars


def contains_by_id(idx: int) -> bool:
    return idx in dynamic_vars


def mark_dynamic_pc(frame_id: int, pc: int, dyn: Dynamic) -> None:
    dynamic_pcs[(frame_id, pc)] = dyn


def contains_pc(frame_id: int, pc: int) -> bool:
    return (frame_id, pc) in dynamic_pcs


def pop_dynamic_pc(frame_id: int, pc: int) -> Dynamic:
    return dynamic_pcs.pop((frame_id, pc))


def add_branch_rewrite_pc(frame_id: int, pc: int) -> None:
    if frame_id not in dynamic_need_branch_rewrite:
        dynamic_need_branch_rewrite[frame_id] = list()
    dynamic_need_branch_rewrite[frame_id].append(pc)


def need_branch_rewrite(frame_id: int) -> bool:
    return frame_id in dynamic_need_branch_rewrite


def get_branch_rewrite_pcs(frame_id: int) -> list[int]:
    return dynamic_need_branch_rewrite[frame_id]


def reset() -> None:
    dynamic_vars.clear()
    dynamic_refs.clear()
    dynamic_pcs.clear()
