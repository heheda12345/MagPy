from frontend.c_api import set_eval_frame, set_skip_files, get_value_stack_from_top, guard_match
from frontend.bytecode_writter import rewrite_bytecode
import dis
import sys
import traceback


# https://docs.python.org/3/library/sys.html#sys.settrace
# a global tracing function must have been installed with settrace() in order to enable assigning frame.f_trace
def simple_trace_func(frame, event, arg):
    return None


def check_fn(locals):
    print("running check_fn, locals:", locals)
    return locals['b'] == 2


def graph_fn():
    return 3


def preprocess_frame(frame, frame_id):
    try:
        print(f"preprocess frame {frame.f_code.co_filename}")
        new_code = rewrite_bytecode(frame.f_code)
        # sys.settrace(simple_trace_func)
    except Exception as e:
        print("exception in preprocess:", e, type(e))
        print(traceback.format_exc())
        raise e
    return (new_code, check_fn, graph_fn)


def postprocess_frame(frame):
    try:
        print(f"postprocess frame {frame.f_code.co_filename}")
        # sys.settrace(None)
        # print("bytecode", list(dis.get_instructions(frame.f_code)))
    except Exception as e:
        print(e)


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


def trace_func(frame, event, arg):
    print(f"trace_func {frame.f_code.co_filename} {event} {arg} {id(frame)}")
    if event == "return":
        print(f"trace_func: return value is {arg}")
    elif event == "opcode":
        global last_op_code
        if last_op_code in LOAD_OPCODES:
            obj = get_value_stack_from_top(frame, 0)
            print("obj", obj, type(obj))
        opcode = frame.f_code.co_code[frame.f_lasti]
        last_op_code = opcode
        if opcode in LOAD_OPCODES:
            opname = dis.opname[opcode]
            print("opname", opname)
        elif opcode in STORE_OPCODES:
            print(f"trace_func: opcode is {dis.opname[opcode]}")


def run_graph(graph_id, *args, **kwargs):
    print("run_graph", graph_id, args, kwargs)
    return None


init = False


def compile(f):
    global init
    if not init:
        set_skip_files(set())
        init = True
        import builtins
        setattr(builtins, "guard_match", guard_match)

    def _fn(*args, **kwargs):
        prior = set_eval_frame(
            (preprocess_frame, postprocess_frame, trace_func))
        try:
            return f(*args, **kwargs)
        except Exception as e:
            print("exception in _fn:", e, type(e))
            raise e
        finally:
            print("restoring frame, prior =", prior)
            set_eval_frame(prior)

    return _fn
