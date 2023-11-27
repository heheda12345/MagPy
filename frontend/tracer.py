import sys
import dis
import traceback
from types import FrameType, CodeType
from typing import Any, Callable, Tuple
import inspect
from .guard_tracker import push_tracker, pop_tracker, record
from .cache import enable_cache, check_cache_updated, get_frame_cache
from .fx_graph import set_frame_root
from .c_api import set_eval_frame, mark_need_postprocess
from .code import ProcessedCode
from .instruction import format_insts
from .config import get_config


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
            elif event == "line":
                print(
                    f"tracing {event} {frame.f_code.co_filename}:{frame.f_lineno}"
                )
            else:
                print(f"tracing {event} in {frame.f_code.co_filename}")
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


def get_process_frame(
        f: Callable[..., Any],
        is_callee: bool) -> Tuple[Callable[..., Any], Callable[..., Any]]:

    is_debug = get_config('debug')

    def preprocess_frame(
            frame: FrameType, frame_id: int
    ) -> Tuple[CodeType, Callable[..., Any], ProcessedCode]:
        try:
            if is_debug:
                print(f"preprocess frame {frame.f_code.co_filename}", frame_id,
                      hex(id(frame)), frame.f_code.co_name)
            enable_cache(frame_id)
            set_frame_root(frame_id, f)
            frame_cache = get_frame_cache(frame_id)
            frame_cache.update_code(frame.f_code, frame_id, is_callee)
            new_code, code_map = frame_cache.get_new_code(is_callee)
            print("bytecode to run:")
            print(format_insts(code_map.guard_insts))
            trace_func = get_trace_func(frame_id)

        except Exception as e:
            print("exception in preprocess:", e, type(e))
            print(traceback.format_exc())
            raise e
        return (new_code, trace_func, code_map)

    def postprocess_frame(frame: FrameType, frame_id: int) -> None:
        try:
            from .bytecode_writter import SHOULD_NOT_CALL_REWRITE
            if SHOULD_NOT_CALL_REWRITE:
                raise ValueError("should not call postprocess")
            print(f"postprocess frame {frame.f_code.co_filename}")
            set_frame_root(frame_id, f)
            frame_cache = get_frame_cache(frame_id)
            frame_cache.update_code(frame.f_code, frame_id, is_callee)
        except Exception as e:
            print("exception in postprocess:", e, type(e))
            print(traceback.format_exc())
            raise e
        return

    return (preprocess_frame, postprocess_frame)