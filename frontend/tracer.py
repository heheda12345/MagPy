import sys
import dis
import traceback
from types import FrameType
from typing import Any, Callable
from frontend.frame_saver import load_frame
from frontend.guard_tracker import push_tracker, pop_tracker, record
import inspect


def get_trace_func(frame_id: int) -> Callable[[FrameType, str, Any], None]:

    def trace_func(frame: FrameType, event: str, arg: Any) -> None:
        try:
            if event == "opcode":
                opcode = frame.f_code.co_code[frame.f_lasti]
                opname = dis.opname[opcode]
                print(
                    f"tracing {event} {opname} {arg} pc={frame.f_lasti} frame={frame_id}({hex(id(frame))})"
                )
                record(frame, frame_id)
            else:
                print(f"tracing {event} {arg} in {frame.f_code.co_filename}")
        except Exception as e:
            print("exception in trace_func:", e, type(e))
            print(traceback.format_exc())
            raise e
        return None

    return trace_func


def empty_trace_func(_frame: FrameType, _event: str, _arg: Any) -> None:
    return None


def enable_trace(frame_id: int) -> None:
    try:
        print("enable_trace")
        this_frame = inspect.currentframe()
        assert this_frame is not None
        caller_frame = this_frame.f_back
        assert caller_frame is not None
        push_tracker(caller_frame, frame_id)
        sys.settrace(empty_trace_func)
    except Exception as e:
        print("exception in enable_trace:", e, type(e))
        print(traceback.format_exc())
        raise e


def disable_trace(frame_id: int) -> None:
    try:
        print("disable_trace")
        pop_tracker(frame_id)
        sys.settrace(None)
    except Exception as e:
        print("exception in disable_trace:", e, type(e))
        print(traceback.format_exc())
        raise e
