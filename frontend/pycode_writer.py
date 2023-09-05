from typing import Any
import struct
import keyword


def get_float_string(value: float) -> str:
    binary_data = struct.pack('d', value)
    hex_string = "b'" + ''.join(
        '\\x' + format(byte, '02x') for byte in binary_data) + "'"
    return f"struct.unpack('d', {hex_string})[0]"


NEW_VAR_ID = 0


def new_name(prefix: str) -> str:
    if prefix == "":
        prefix = "tmp"
    global NEW_VAR_ID
    NEW_VAR_ID += 1
    return f"{prefix}_{NEW_VAR_ID}"


def is_valid_name(variable_name: str) -> bool:
    if not variable_name:
        return False

    if not variable_name[0].isalpha() and variable_name[0] != '_':
        return False

    for char in variable_name[1:]:
        if not (char.isalnum() or char == '_'):
            return False

    if keyword.iskeyword(variable_name):
        return False

    return True


class PyCodeWriter:
    imports: set[str]
    code_strs: list[str]
    indent: int

    def __init__(self) -> None:
        self.imports = set()
        self.code_strs = []
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
            self.wl(line)

    def wl(self, code_str: str) -> None:
        if code_str.endswith('\n'):
            code_str = code_str[:-1]
        self.code_strs.append('    ' * self.indent + code_str)

    def get_code(self) -> str:
        return '\n'.join(self.code_strs)
