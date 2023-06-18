import sys
import dis
import traceback
from types import FrameType
from typing import Any, Callable


def get_trace_func(frame_id: int) -> Callable[[FrameType, str, Any], None]:

    def trace_func(frame: FrameType, event: str, arg: Any) -> None:
        try:
            print(
                f"simple_trace_func {frame.f_code.co_filename} {event} {arg} {id(frame)} frame_id={frame_id}"
            )
            if event == "opcode":
                opcode = frame.f_code.co_code[frame.f_lasti]
                opname = dis.opname[opcode]
                print("opname", opname)
        except Exception as e:
            print("exception in simple_trace:", e, type(e))
            print(traceback.format_exc())
            raise e
        return None

    return trace_func


def empty_trace_func(_frame: FrameType, _event: str, _arg: Any) -> None:
    return None


def enable_trace() -> None:
    print("enable_trace")
    sys.settrace(empty_trace_func)


def disable_trace() -> None:
    print("disable_trace")
    sys.settrace(None)
