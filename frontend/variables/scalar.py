from typing import Any
from .base import Guard
from ..utils import get_float_string


class ScalarGuard(Guard):

    def __init__(self, extract_code: str, value: Any) -> None:
        if type(value) == float:
            super().__init__([f"{extract_code} == {get_float_string(value)}"],
                             set(["struct"]))
        else:
            super().__init__([f"{extract_code} == {value}"])
