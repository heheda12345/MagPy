from typing import Any, Callable, Dict, Optional, Tuple
from types import FrameType


def set_eval_frame(
    new_callback: Tuple[Callable[..., Any], Callable[..., Any]]
) -> Tuple[Callable[..., Any], Callable[..., Any]]:
    pass


def set_skip_files(skip_file: set[str]) -> None:
    pass


def get_value_stack_from_top(frame: FrameType, index: int) -> list[Any]:
    pass


def guard_match(frame_id: int, callsite_id: int,
                locals: Dict[str, Any]) -> Optional[Callable[..., Any]]:
    pass

def finalize() -> None:
    pass