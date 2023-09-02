from .base import Guard
from typing import Any


class ScalarGuard(Guard):

    def __init__(self, extract_code: str, value: Any) -> None:
        super().__init__([f"{extract_code} == {value}"])
