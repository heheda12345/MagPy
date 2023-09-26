from typing import Any


class StorePos:
    pass


class StoreInStack(StorePos):
    idx: int

    def __init__(self, idx: int) -> None:
        self.idx = idx

    def __str__(self) -> str:
        return f"__stack__{self.idx}"


class StoreInLocal(StorePos):
    name: str

    def __init__(self, name: str) -> None:
        self.name = name

    def __str__(self) -> str:
        return f"locals['{self.name}']"


class StoreInGlobal(StorePos):
    name: str

    def __init__(self, name: str) -> None:
        self.name = name

    def __str__(self) -> str:
        return f"globals()['{self.name}']"


class StoreInAttr(StorePos):
    self_pos: StorePos
    self_obj: Any
    attr_name: str

    def __init__(self, self_pos: StorePos, self_obj: Any,
                 attr_name: str) -> None:
        self.self_pos = self_pos
        self.self_obj = self_obj
        self.attr_name = attr_name

    def __str__(self) -> str:
        return f"{self.self_pos}.{self.attr_name}"


class StoreInIndex(StorePos):
    self_pos: StorePos
    self_idx: Any
    subscriptable: bool

    def __init__(self,
                 self_pos: StorePos,
                 self_idx: Any,
                 subscritable: bool = True) -> None:
        self.self_pos = self_pos
        self.self_idx = self_idx
        self.subscriptable = subscritable

    def __str__(self) -> str:
        if self.subscriptable:
            return f"{self.self_pos}[{self.self_idx}]"
        else:
            return f'list({self.self_pos})[{self.self_idx}]'