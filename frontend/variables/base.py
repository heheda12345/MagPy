from dataclasses import dataclass
from ..instruction import Instruction, ci


@dataclass
class Guard:
    code: list[str]


@dataclass
class Variable:
    guard: Guard
    extract_code: str
    extract_insts: list[Instruction]


class RuntimeVar(Variable):

    def __init__(self) -> None:
        super().__init__(Guard([]),
                         "@@RUNTIME_VAR, should not read this field@@", [])


class StackVar(Variable):
    depth: int

    def __init__(self, depth: int) -> None:
        var_name = f"__stack__{depth}"
        super().__init__(Guard([]), f"locals['{var_name}']",
                         [ci('LOAD_FAST', var_name, var_name)])
