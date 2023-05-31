from frontend.c_api import set_eval_frame, set_skip_files, get_value_stack_from_top
import dis
import sys

# https://docs.python.org/3/library/sys.html#sys.settrace
# a global tracing function must have been installed with settrace() in order to enable assigning frame.f_trace
def simple_trace_func(frame, event, arg):
    return None

def preprocess_frame(frame):
    try:
        print(f"preprocess frame {frame.f_code.co_filename}", id(frame))
        print("bytecode", list(dis.get_instructions(frame.f_code)))
        sys.settrace(simple_trace_func)
    except Exception as e:
        print(e)

def postprocess_frame(frame):
    try:
        print(f"postprocess frame {frame.f_code.co_filename}")
        sys.settrace(None)
        # print("bytecode", list(dis.get_instructions(frame.f_code)))
    except Exception as e:
        print(e)



LOAD_OPCODES = list(map(dis.opmap.get, ["LOAD_GLOBAL", "LOAD_NAME", "LOAD_FAST", "LOAD_DEREF", "LOAD_ASSERTION_ERROR", "LOAD_BUILD_CLASS", "LOAD_CONST", "LOAD_ATTR", "LOAD_CLOSURE", "LOAD_CLASSDEREF", "LOAD_METHOD"]))
STORE_OPCODES = list(map(dis.opmap.get, ["STORE_SUBSCR", "STORE_NAME", "STORE_ATTR", "STORE_GLOBAL", "STORE_FAST", "STORE_DEREF"]))

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


def compile(f):
    if not hasattr(compile, "skip_file_setted"):
        set_skip_files(set())
        compile.skip_file_setted = True
    def _fn(*args, **kwargs):
        prior = set_eval_frame((preprocess_frame, postprocess_frame, trace_func))
        try:
            return f(*args, **kwargs)
        except Exception as e:
            print(e)
        finally:
            print("restoring frame")
            print("prior:", prior)
            set_eval_frame(prior)
    return _fn
