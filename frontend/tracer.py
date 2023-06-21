import sys
import dis
import traceback
from types import FrameType
from typing import Any, Callable
from frontend.frame_saver import load_frame
from frontend.guard_tracker import push_tracker, pop_tracker, record


def get_trace_func(frame_id: int) -> Callable[[FrameType, str, Any], None]:

    def trace_func(frame: FrameType, event: str, arg: Any) -> None:
        try:
            if event == "opcode":
                opcode = frame.f_code.co_code[frame.f_lasti]
                opname = dis.opname[opcode]
                print(
                    f"trace_func {frame.f_code.co_filename} {event} {arg} {id(frame)} frame_id={frame_id} opname={opname}"
                )
                record(frame, frame_id)
            else:
                print(f"trace_func {frame.f_code.co_filename} {event} {arg}")
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
        push_tracker(frame_id)
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
