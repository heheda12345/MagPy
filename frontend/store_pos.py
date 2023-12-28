from typing import Any, Optional, TYPE_CHECKING, Callable
from types import FrameType

from .c_api import get_value_stack_from_top
if TYPE_CHECKING:
    from .pycode_generator import FnCodegen


class StorePos:

    def get_value_from_frame(self, frame: FrameType) -> Any:
        raise NotImplementedError

    def add_name_to_fn(self, codegen: 'FnCodegen') -> None:
        pass


class StoreInStack(StorePos):
    idx: int

    def __init__(self, idx: int) -> None:
        self.idx = idx

    def __repr__(self) -> str:
        return f"__stack__{self.idx}"

    def get_value_from_frame(self, frame: FrameType) -> Any:
        return get_value_stack_from_top(frame, self.idx)


class StoreInLocal(StorePos):
    name: str

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f"locals['{self.name}']"

    def get_value_from_frame(self, frame: FrameType) -> Any:
        return frame.f_locals[self.name]


class StoreInGlobal(StorePos):
    name: str

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f"globals()['{self.name}']"

    def get_value_from_frame(self, frame: FrameType) -> Any:
        return frame.f_globals[self.name]


class StoreInFreeVar(StorePos):
    free_idx: int

    def __init__(self, free_idx: int) -> None:
        super().__init__()
        self.free_idx = free_idx

    def __repr__(self) -> str:
        return f"get_from_freevars(frame, {self.free_idx})"

    def get_value_from_frame(self, frame: FrameType) -> Any:
        raise NotImplementedError

    def add_name_to_fn(self, codegen: 'FnCodegen') -> None:
        codegen.add_import_from("frontend.c_api", "get_from_freevars")
        codegen.add_import("inspect")
        codegen.add_stmt("frame = inspect.currentframe().f_back")


class StoreInBuiltin(StorePos):
    name: str
    ty: str  # attr or dict

    def __init__(self, name: str, ty: str) -> None:
        self.name = name
        self.ty = ty
        assert ty in ['attr', 'dict']

    def __repr__(self) -> str:
        if self.ty == 'dict':
            return f"globals()['__builtins__']['{self.name}']"
        else:
            return f"globals()['__builtins__'].{self.name}"

    def get_value_from_frame(self, frame: FrameType) -> Any:
        if self.ty == 'dict':
            return frame.f_globals['__builtins__'][self.name]
        else:
            return getattr(frame.f_globals['__builtins__'], self.name)


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

    def get_value_from_frame(self, frame: FrameType) -> Any:
        return getattr(self.self_pos.get_value_from_frame(frame),
                       self.attr_name)

    def add_name_to_fn(self, codegen: 'FnCodegen') -> None:
        self.self_pos.add_name_to_fn(codegen)


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

    def __repr__(self) -> str:
        if self.subscriptable:
            return f"{self.self_pos}[{self.self_index}]"
        else:
            return f'list({self.self_pos})[{self.self_index}]'

    def get_value_from_frame(self, frame: FrameType) -> Any:
        if self.subscriptable:
            return self.self_pos.get_value_from_frame(frame)[self.self_index]
        else:
            return list(
                self.self_pos.get_value_from_frame(frame))[self.self_index]

    def add_name_to_fn(self, codegen: 'FnCodegen') -> None:
        self.self_pos.add_name_to_fn(codegen)


class StoreNegate(StorePos):
    pos: StorePos
    neg_id: int

    def __init__(self, pos: StorePos) -> None:
        self.pos = pos

    def __repr__(self) -> str:
        return f"-({self.pos})"

    def get_value_from_frame(self, frame: FrameType) -> Any:
        return -self.pos.get_value_from_frame(frame)

    def add_name_to_fn(self, codegen: 'FnCodegen') -> None:
        self.pos.add_name_to_fn(codegen)


class ExtractFromMethod(StorePos):
    self_pos: StorePos
    self_id: int
    method_name: str

    def __init__(self, self_pos: StorePos, self_id: int,
                 method_name: str) -> None:
        self.self_pos = self_pos
        self.self_id = self_id
        self.method_name = method_name

    def __repr__(self) -> str:
        return f"{self.self_pos}.{self.method_name}()"

    def get_value_from_frame(self, frame: FrameType) -> Any:
        return getattr(self.self_pos.get_value_from_frame(frame),
                       self.method_name)()

    def add_name_to_fn(self, codegen: 'FnCodegen') -> None:
        self.self_pos.add_name_to_fn(codegen)


class ExtractFromFunction(StorePos):
    var_pos: list[StorePos]
    var_id: list[int]
    func_name: str
    func_obj: Any
    need_add_to_fn: bool
    preserved_name: str

    def __init__(self,
                 var_pos: list[StorePos],
                 var_id: list[int],
                 func_name: str,
                 func_obj: Callable[..., Any],
                 need_add_to_fn: bool = False) -> None:
        self.var_pos = var_pos
        self.var_id = var_id
        self.func_name = func_name
        self.func_obj = func_obj
        self.need_add_to_fn = need_add_to_fn
        from .pycode_writer import new_name
        self.preserved_name = new_name(f"function_{self.func_name}")

    def __repr__(self) -> str:
        if self.need_add_to_fn:
            return f"{self.preserved_name}({','.join([str(pos) for pos in self.var_pos])})"
        else:
            return f"{self.func_name}({','.join([str(pos) for pos in self.var_pos])})"

    def get_value_from_frame(self, frame: FrameType) -> Any:
        args = [pos.get_value_from_frame(frame) for pos in self.var_pos]
        return self.func_obj(*args)

    def add_name_to_fn(self, codegen: 'FnCodegen') -> None:
        if self.need_add_to_fn:
            codegen.add_obj(self.func_obj, self.preserved_name, force=True)
            for pos in self.var_pos:
                pos.add_name_to_fn(codegen)


class ExtractFromNew(StorePos):
    type_obj: Any
    preserved_name: Optional[str]

    def __init__(self, type_obj: Any) -> None:
        self.type_obj = type_obj
        self.preserved_name = None

    def gen_preserved_name(self) -> str:
        if self.preserved_name is None:
            from .pycode_writer import new_name
            self.preserved_name = new_name(f"class_{self.type_obj.__name__}")
        return self.preserved_name

    def __repr__(self) -> str:
        return f"object.__new__({self.gen_preserved_name()})"

    def get_value_from_frame(self, frame: FrameType) -> Any:
        return self.type_obj.__new__(self.type_obj)

    def add_name_to_fn(self, codegen: 'FnCodegen') -> None:
        codegen.add_obj(self.type_obj, self.gen_preserved_name(), force=True)


class IterValue(StorePos):

    def __init__(self) -> None:
        super().__init__()

    def __repr__(self) -> str:
        return "__iter_value__"


class UnknownPosInCaller(StorePos):

    def __init__(self) -> None:
        super().__init__()

    def __repr__(self) -> str:
        return "@__unknown_pos_in_caller__"


class voidpos(StorePos):
    pass