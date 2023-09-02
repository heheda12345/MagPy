import inspect
import dis
from typing import Any
import struct
from .bytecode_writter import get_code_keys


def print_bytecode() -> None:
    this_frame = inspect.currentframe()  # the print_bytecode function
    assert this_frame is not None
    test_func_frame = this_frame.f_back
    assert test_func_frame is not None
    code = test_func_frame.f_code
    insts = dis.Bytecode(code)
    for inst in insts:
        print(inst)
    keys = get_code_keys()
    code_options = {k: getattr(code, k) for k in keys}
    for k, v in code_options.items():
        print(k, v)


class PyCodeWriter:
    imports: set[str]
    code_str: str
    indent: int

    def __init__(self) -> None:
        self.imports = set()
        self.code_str = ''
        self.indent = 0

    def block_start(self) -> None:
        self.indent += 1

    def block_end(self) -> None:
        self.indent -= 1

    def set_indent(self, indent: int) -> None:
        self.indent = indent

    def write(self, code_str: str) -> None:
        print("writing: indent", self.indent)
        code = code_str.splitlines()
        for line in code:
            self.code_str += '    ' * self.indent + line + '\n'

    def wl(self, code_str: str) -> None:
        self.write(code_str + '\n')

    def get_code(self) -> str:
        return self.code_str

    def add_import(self, module_name: str) -> None:
        self.imports.add(module_name)

    def get_imports(self, indent: int) -> str:
        print("imports:", self.imports)
        return '\n'.join(f'{"    " * indent}import {module_name}'
                         for module_name in self.imports)


def is_scalar(value: Any) -> bool:
    return type(value) in {int, float, bool, str}


def get_float_string(value: float) -> str:
    binary_data = struct.pack('d', value)
    hex_string = "b'" + ''.join(
        '\\x' + format(byte, '02x') for byte in binary_data) + "'"
    return f"struct.unpack('d', {hex_string})[0]"