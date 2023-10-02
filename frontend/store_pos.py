from typing import Any


class StorePos:
    pass


class StoreInStack(StorePos):
    idx: int

    def __init__(self, idx: int) -> None:
        self.idx = idx

    def __repr__(self) -> str:
        return f"__stack__{self.idx}"


class StoreInLocal(StorePos):
    name: str

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f"locals['{self.name}']"


class StoreInGlobal(StorePos):
    name: str

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f"globals()['{self.name}']"


class StoreInAttr(StorePos):
    self_pos: StorePos
    self_id: int
    attr_name: str

    def __init__(self, self_pos: StorePos, self_id: int,
                 attr_name: str) -> None:
        self.self_pos = self_pos
        self.self_id = self_id
        self.attr_name = attr_name

    def __repr__(self) -> str:
        return f"{self.self_pos}.{self.attr_name}"


class StoreInIndex(StorePos):
    self_pos: StorePos
    self_id: int  # id of the bind object
    self_index: Any  # array index
    subscriptable: bool

    def __init__(self,
                 self_pos: StorePos,
                 self_id: int,
                 self_index: Any,
                 subscritable: bool = True) -> None:
        self.self_pos = self_pos
        self.self_id = self_id
        self.self_index = self_index
        self.subscriptable = subscritable

    def __str__(self) -> str:
        if self.subscriptable:
            return f"{self.self_pos}[{self.self_index}]"
        else:
            return f'list({self.self_pos})[{self.self_index}]'