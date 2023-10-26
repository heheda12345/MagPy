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
from .bytecode_writter import rewrite_bytecode
from .code import ProcessedCode
from .instruction import format_insts


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

    def preprocess_frame(
            frame: FrameType, frame_id: int
    ) -> Tuple[CodeType, Callable[..., Any], ProcessedCode]:
        try:
            print(f"preprocess frame {frame.f_code.co_filename}", frame_id,
                  hex(id(frame)), frame.f_code.co_name)
            enable_cache(frame_id)

            if get_frame_cache(frame_id).pre_cache_size == -1:
                print("new bytecode: \n")
                set_frame_root(frame_id, f)
                new_code, code_map = rewrite_bytecode(frame.f_code, frame_id,
                                                    is_callee)
                get_frame_cache(frame_id).new_code = new_code
                get_frame_cache(frame_id).code_map = code_map
                trace_func = get_trace_func(frame_id)
            else:
                print("old bytecode: \n")
                print(format_insts(get_frame_cache(frame_id).code_map.guard_insts))
                new_code = get_frame_cache(frame_id).new_code
                code_map = get_frame_cache(frame_id).code_map
                trace_func = get_trace_func(frame_id)
                mark_need_postprocess()                
        except Exception as e:
            print("exception in preprocess:", e, type(e))
            print(traceback.format_exc())
            raise e
        return (new_code, trace_func, code_map)

    def postprocess_frame(frame: FrameType, frame_id: int) -> None:
        try:
            print(f"postprocess frame {frame.f_code.co_filename}")
            if check_cache_updated(frame_id):
                print("new bytecode: \n")
                set_frame_root(frame_id, f)
                new_code, code_map = rewrite_bytecode(frame.f_code, frame_id,
                                                    is_callee)
                trace_func = get_trace_func(frame_id)

        except Exception as e:
            print("exception in postprocess:", e, type(e))
            print(traceback.format_exc())
            raise e
        return (new_code, trace_func, code_map)

    return (preprocess_frame, postprocess_frame)