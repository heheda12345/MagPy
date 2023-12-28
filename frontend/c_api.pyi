from typing import Any, Callable, Dict, Optional, Tuple, TYPE_CHECKING, Iterable
from types import FrameType, CellType

if TYPE_CHECKING:
    from .code import ProcessedCode


def set_eval_frame(
    new_callback: Optional[Tuple[Callable[..., Any], Callable[..., Any]]]
) -> Optional[Tuple[Callable[..., Any], Callable[..., Any]]]:
    pass


def set_skip_files(skip_file: set[str], end_file: set[str]) -> None:
    pass


def get_value_stack_from_top(frame: FrameType, index: int) -> Any:
    pass


def set_value_stack_from_top(frame: FrameType, index: int, value: Any) -> None:
    pass


def get_value_stack_size(frame: FrameType) -> int:
    pass


def guard_match(frame_id: int, callsite_id: int,
                locals: Dict[str, Any]) -> Optional[Callable[..., Any]]:
    pass

def get_miss_locals(frame_id: int) -> list[str]:
    pass

def finalize() -> None:
    pass


def enter_nested_tracer() -> None:
    pass


def exit_nested_tracer() -> None:
    pass


def mark_need_postprocess() -> None:
    pass


def add_to_cache(frame_id: int, callsite_id: int, id_in_callsite: int,
                 guard_fn: Callable[..., Any], graph_fn: Callable[...,
                                                                  Any]) -> None:
    pass


def c_reset() -> None:
    pass


def stack_effect(op: int, oparg: int,
                 jump: Optional[bool]) -> tuple[int, int, int, bool, bool]:
    pass


def set_null_object(obj: Any) -> None:
    pass


def set_miss_threshold(obj: Any) -> None:
    pass


def get_next_frame_id() -> int:
    pass


def get_code_map(frame: FrameType) -> 'ProcessedCode':
    pass


def is_bound_method(obj: Any, name: str) -> bool:
    pass


def parse_rangeiterobject(obj: Any) -> Tuple[int, int, int, int]:
    pass


def parse_mapproxyobject(obj: Any) -> Any:
    pass


def make_rangeiterobject(start: int, stop: int, step: int) -> Any:
    pass


def get_from_freevars(frame: FrameType, idx: int) -> Any:
    pass


def parse_mapobject(obj: Any) -> Tuple[Iterable[Any], Callable[..., Any]]:
    pass

def parse_cell(cell: CellType) -> Any:
    pass

def set_cell(cell: CellType, value: Any) -> None:
    pass

def set_local(frame: FrameType, idx: int, value: Any) -> None:
    pass
