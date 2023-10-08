from typing import Any, Callable, Dict, Optional, Tuple, TYPE_CHECKING
from types import FrameType

if TYPE_CHECKING:
    from .code import ProcessedCode


def set_eval_frame(
    new_callback: Optional[Tuple[Callable[..., Any], Callable[..., Any]]]
) -> Optional[Tuple[Callable[..., Any], Callable[..., Any]]]:
    pass


def set_skip_files(skip_file: set[str]) -> None:
    pass


def get_value_stack_from_top(frame: FrameType, index: int) -> Any:
    pass


def get_value_stack_size(frame: FrameType) -> int:
    pass


def guard_match(frame_id: int, callsite_id: int,
                locals: Dict[str, Any]) -> Optional[Callable[..., Any]]:
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


def get_next_frame_id() -> int:
    pass


def get_code_map(frame: FrameType) -> 'ProcessedCode':
    pass


def is_bound_method(obj: Any, name: str) -> bool:
    pass