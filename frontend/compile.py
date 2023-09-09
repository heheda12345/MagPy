import dis
import sys
import traceback
from types import FrameType, CodeType
from typing import Any, Tuple, Callable
import logging
import torch
from .c_api import set_eval_frame, set_skip_files, guard_match, c_reset, set_null_object
from .bytecode_writter import rewrite_bytecode
from .tracer import enable_trace, disable_trace, get_trace_func
from .cache import enable_cache
from .utils import null_object

logging.basicConfig(
    format='%(levelname)s [%(filename)s:%(lineno)d] %(message)s',
    level=logging.INFO)


def preprocess_frame(frame: FrameType,
                     frame_id: int) -> Tuple[CodeType, Callable[..., Any]]:
    try:
        print(f"preprocess frame {frame.f_code.co_filename}", frame_id,
              hex(id(frame)))
        enable_cache(frame_id)
        new_code = rewrite_bytecode(frame.f_code, frame_id)
        trace_func = get_trace_func(frame_id)
    except Exception as e:
        print("exception in preprocess:", e, type(e))
        print(traceback.format_exc())
        raise e
    return (new_code, trace_func)


def postprocess_frame(frame: FrameType) -> None:
    try:
        print(f"postprocess frame {frame.f_code.co_filename}")
    except Exception as e:
        print("exception in postprocess:", e, type(e))
        print(traceback.format_exc())
        raise e
    return None


LOAD_OPCODES = list(
    map(dis.opmap.get, [
        "LOAD_GLOBAL", "LOAD_NAME", "LOAD_FAST", "LOAD_DEREF",
        "LOAD_ASSERTION_ERROR", "LOAD_BUILD_CLASS", "LOAD_CONST", "LOAD_ATTR",
        "LOAD_CLOSURE", "LOAD_CLASSDEREF", "LOAD_METHOD"
    ]))
STORE_OPCODES = list(
    map(dis.opmap.get, [
        "STORE_SUBSCR", "STORE_NAME", "STORE_ATTR", "STORE_GLOBAL",
        "STORE_FAST", "STORE_DEREF"
    ]))

last_op_code = dis.opmap.get("NOP")


def run_graph(graph_id: int, *args: Any, **kwargs: Any) -> None:
    print("run_graph", graph_id, args, kwargs)
    return None


init = False


def compile(f: Callable[..., Any]) -> Callable[..., Any]:
    global init
    if not init:
        set_skip_files(set())
        set_null_object(null_object)
        init = True
        import builtins
        setattr(builtins, "guard_match", guard_match)
        setattr(builtins, "enable_trace", enable_trace)
        setattr(builtins, "disable_trace", disable_trace)

    def _fn(*args: Any, **kwargs: Any) -> Any:
        prior = set_eval_frame((preprocess_frame, postprocess_frame))
        try:
            fn = f.forward if isinstance(f, torch.nn.Module) else f
            return fn(*args, **kwargs)
        except Exception as e:
            print("exception in _fn:", e, type(e))
            raise e
        finally:
            print("restoring frame, prior =", prior)
            set_eval_frame(prior)

    return _fn


def reset() -> None:
    c_reset()
    from . import code
    code.reset()
    from . import cache
    cache.reset()
    from . import guard_tracker
    guard_tracker.reset()
    from . import utils
    utils.reset()