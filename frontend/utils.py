import inspect
import dis
from frontend.bytecode_writter import get_code_keys


def print_bytecode():
    test_func_frame = inspect.currentframe().f_back
    code = test_func_frame.f_code
    insts = dis.Bytecode(code)
    for inst in insts:
        print(inst)
    keys = get_code_keys()
    code_options = {k: getattr(code, k) for k in keys}
    for k, v in code_options.items():
        print(k, v)
