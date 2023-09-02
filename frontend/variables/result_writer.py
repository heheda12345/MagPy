from typing import Any
from ..utils import PyCodeWriter


class ResultWriter:
    writer: PyCodeWriter

    def __init__(self, initial_indent: int = 0) -> None:
        self.writer = PyCodeWriter()
        self.writer.intend = initial_indent

    def save(self, target_name: str, var: Any) -> None:
        if type(var) in {int, float, bool, str}:
            self.save_scalar(target_name, var)
        else:
            raise NotImplementedError(f"unknown type in reproduce: {type(var)}")

    def save_scalar(self, target_name: str, var: Any) -> None:
        self.writer.write(f"{target_name} = {var}")

    def get_code(self) -> str:
        return self.writer.get_code()
