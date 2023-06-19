from typing import Any, Optional, Union
import dataclasses
import dis


@dataclasses.dataclass
class Instruction:
    """A mutable version of dis.Instruction"""

    opcode: int
    opname: str
    arg: Any
    argval: Any
    offset: Optional[int] = None
    starts_line: Optional[int] = None
    is_jump_target: bool = False
    # extra fields to make modification easier:
    target: Optional["Instruction"] = None
    original_inst: Optional["Instruction"] = None

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:
        return id(self) == id(other)


def convert_instruction(i: dis.Instruction) -> Instruction:
    return Instruction(
        i.opcode,
        i.opname,
        i.arg,
        i.argval,
        i.offset,
        i.starts_line,
        i.is_jump_target,
    )


class _NotProvided:
    pass


# short for create_instruction
def ci(name: str,
       arg: Any = None,
       argval: Any = _NotProvided,
       target: Optional[Instruction] = None) -> Instruction:
    if argval is _NotProvided:
        argval = arg
    return Instruction(opcode=dis.opmap[name],
                       opname=name,
                       arg=arg,
                       argval=argval,
                       target=target)
