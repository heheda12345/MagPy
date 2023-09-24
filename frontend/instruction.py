from typing import Any, Optional
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
    comment: str = ""
    is_start: bool = False
    is_end: bool = False

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:
        return id(self) == id(other)

    def __repr__(self) -> str:
        # yellow if is original inst, green if is generated inst
        color = "\033[33m" if self.original_inst else "\033[32m"
        color_gray = "\033[90m"
        comment = f"{color_gray}# {self.comment} \033[0m" if self.comment else ""
        return f"{color}{self.opname}\033[0m({self.arg}, {self.argval}) {comment}"


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


def format_insts(insts: list[Instruction]) -> str:
    ret = ""
    for i, inst in enumerate(insts):
        if inst.target is not None:
            target_idx = insts.index(inst.target)
            ret += f"{i}: {inst} -> inst {target_idx}\n"
        else:
            ret += f"{i}: {inst}\n"
    return ret


class _NotProvided:
    pass


# short for create_instruction
def ci(name: str,
       arg: Any = None,
       argval: Any = _NotProvided,
       target: Optional[Instruction] = None,
       comment: str = "") -> Instruction:
    if argval is _NotProvided:
        argval = arg
    return Instruction(opcode=dis.opmap[name],
                       opname=name,
                       arg=arg,
                       argval=argval,
                       target=target,
                       comment=comment)
