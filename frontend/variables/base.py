from dataclasses import dataclass
from abc import abstractmethod
from typing import Any, TYPE_CHECKING
if TYPE_CHECKING:
    from ..pycode_generator import GraphFnCodegen, GuardFnCodegen


@dataclass
class Variable:
    need_guard_check: bool
    extract_code_at_start: str = ""

    def __init__(self,
                 need_guard_check: bool,
                 extract_code_at_start: str = "") -> None:
        self.need_guard_check = need_guard_check
        self.extract_code_at_start = extract_code_at_start
        if need_guard_check:
            assert extract_code_at_start != ""

    @classmethod
    @abstractmethod
    def from_value(self,
                   value: Any,
                   need_guard_check: bool,
                   extract_code_at_start: str = "") -> 'Variable':
        raise NotImplementedError

    def make_guard(self, writer: GuardFnCodegen) -> None:
        if self.need_guard_check:
            self.make_guard_inner(writer)

    @abstractmethod
    def make_guard_inner(self, writer: GuardFnCodegen) -> None:
        raise NotImplementedError

    @abstractmethod
    def make_output(self, target_name: str, codegen: GraphFnCodegen) -> None:
        raise NotImplementedError
