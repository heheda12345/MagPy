from typing import Any
from ..utils import PyCodeWriter, get_float_string

NEW_VAR_ID = 0


def new_name() -> str:
    global NEW_VAR_ID
    NEW_VAR_ID += 1
    return f"tmp_{NEW_VAR_ID}"


class ResultWriter:
    writer: PyCodeWriter

    def __init__(self, initial_indent: int = 0) -> None:
        self.writer = PyCodeWriter()
        self.writer.set_indent(initial_indent)

    def save(self, target_name: str, var: Any) -> None:
        if type(var) in {int, float, bool, str}:
            self.save_scalar(target_name, var)
        else:
            raise NotImplementedError(f"unknown type in reproduce: {type(var)}")

    def save_scalar(self, target_name: str, var: Any) -> None:
        if type(var) == float:
            self.writer.wl(f"{target_name} = {get_float_string(var)} # {var}")
            self.writer.add_import("struct")
        else:
            self.writer.wl(f"{target_name} = {var}")

    def get_code(self) -> str:
        return self.writer.get_code()

    def get_imports(self, indent) -> str:
        return self.writer.get_imports(indent)
