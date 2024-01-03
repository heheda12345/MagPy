import dis
import sys
import traceback
from types import FrameType, CodeType
from typing import Any, Tuple, Callable, cast
import logging
import inspect
import torch
from . import tracer, utils, guard_tracker
from .config import get_config
from .c_api import set_eval_frame, set_skip_files, guard_match, c_reset, set_null_object, set_miss_threshold
from .tracer import enable_trace, disable_trace, get_trace_func, get_process_frame
from .cache import enable_cache
from .utils import null_object
from .fx_graph import set_frame_root
from .control_flow import if_stmt

logging.basicConfig(
    format='%(levelname)s [%(filename)s:%(lineno)d] %(message)s',
    level=logging.INFO)

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
        nn_module = inspect.getmodule(torch.nn.Module)
        assert nn_module is not None
        set_skip_files(
            set({
                cast(str, nn_module.__file__),
                tracer.__file__,
                utils.__file__,
                torch.autograd.function.__file__,
                torch._functorch.utils.__file__,
            }), set({
                guard_tracker.__file__,
            }))
        set_null_object(null_object)
        set_miss_threshold(get_config("miss_threshold"))
        init = True
        import builtins
        setattr(builtins, "guard_match", guard_match)
        setattr(builtins, "enable_trace", enable_trace)
        setattr(builtins, "disable_trace", disable_trace)
        setattr(builtins, "_frontend_compile_if_stmt", if_stmt)

    def _fn(*args: Any, **kwargs: Any) -> Any:
        pre, post = get_process_frame(f, False)
        prior = set_eval_frame((pre, post))
        try:
            fn = f.forward if isinstance(f, torch.nn.Module) else f
            return fn(*args, **kwargs)
        except Exception as e:
            print("exception in _fn:", e, type(e))
            raise e
        finally:
            set_eval_frame(prior)

    return _fn


def reset() -> None:
    c_reset()
    from . import cache
    cache.reset()
    from . import guard_tracker
    guard_tracker.reset()
    from . import utils
    utils.reset()
    from . import fx_graph
    fx_graph.reset()
    from . import dynamic
    dynamic.reset()
