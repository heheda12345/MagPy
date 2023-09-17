import inspect
import dis
from typing import Any, TYPE_CHECKING, Callable
import random
import operator
from .bytecode_writter import get_code_keys
if TYPE_CHECKING:
    from .instruction import Instruction


class NullObject:
    '''
    The stack should be the following when meth is unbound
    NULL | meth | arg1 | ... | argN
    But as we cannot push NULL into the stack, we push a NullObject instead.
    NullObject | meth | arg1 | ... | argN
    We simulate the behavior of unbound method by calling arg0(arg1, ..., argN)
    '''

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        print("calling unbound method")
        return args[0](*args[1:], **kwargs)


null_object = NullObject()


def print_bytecode() -> None:
    this_frame = inspect.currentframe()  # the print_bytecode function
    assert this_frame is not None
    test_func_frame = this_frame.f_back
    assert test_func_frame is not None
    code = test_func_frame.f_code
    insts = dis.Bytecode(code)
    for inst in insts:
        print(inst)
    keys = get_code_keys()
    code_options = {k: getattr(code, k) for k in keys}
    for k, v in code_options.items():
        print(k, v)


def is_scalar(value: Any) -> bool:
    return type(value) in {int, float, bool, str}


def is_call_bytecode(inst: 'Instruction') -> bool:
    return inst.opname.startswith("CALL_")


fx_graph_functions: set[Callable[..., Any]] = {
    operator.pos,
    operator.neg,
    operator.not_,
    operator.invert,
    operator.pow,
    operator.mul,
    operator.matmul,
    operator.floordiv,
    operator.truediv,
    operator.mod,
    operator.add,
    operator.sub,
    operator.getitem,
    operator.lshift,
    operator.rshift,
    operator.and_,
    operator.or_,
    operator.xor,
    # operator.ipow,
    # operator.imul,
    # operator.imatmul,
    # operator.ifloordiv,
    # operator.itruediv,
    # operator.imod,
    # operator.iadd,
    # operator.isub,
    # operator.ilshift,
    # operator.irshift,
    # operator.iand,
    # operator.ixor,
    # operator.ior,
}


def is_user_defined_func(func: Callable[..., Any]) -> bool:
    if func in fx_graph_functions:
        return False
    module = inspect.getmodule(func)
    if module is None:
        return True
    module_pack = module.__package__
    if module_pack is None:
        return True
    root_module = module_pack.split('.')[0]
    return root_module not in ('math', 'builtins', 'torch', 'numpy')


random_state = None


def new_random_key() -> int:
    global random_state
    cur_state = random.getstate()
    if random_state is None:
        random.seed(23333)
        random_state = random.getstate()
    random.setstate(random_state)
    new_key = random.randint(0, 10000)
    random_state = random.getstate()
    random.setstate(cur_state)
    return new_key


class ForceGraphBreaker:
    breaks: dict[int, set[int]]  # frame_id -> list of pc

    def __init__(self) -> None:
        self.breaks = {}

    def add(self, frame_id: int, pc: int) -> None:
        if frame_id not in self.breaks:
            self.breaks[frame_id] = set()
        self.breaks[frame_id].add(pc)

    def need_break(self, frame_id: int, pc: int) -> bool:
        if frame_id not in self.breaks:
            return False
        return pc in self.breaks[frame_id]


graph_breaker = None


def add_force_graph_break(frame_id: int, pc: int) -> None:
    global graph_breaker
    if graph_breaker is None:
        graph_breaker = ForceGraphBreaker()
    graph_breaker.add(frame_id, pc)


def has_force_graph_break(frame_id: int, pc: int) -> bool:
    global graph_breaker
    # fast path
    if graph_breaker is None:
        return False
    return graph_breaker.need_break(frame_id, pc)


def clear_force_graph_break() -> None:
    global graph_breaker
    graph_breaker = None


class UnknownTypeError(Exception):

    def __init__(self, ty: type[Any]) -> None:
        super().__init__(f"Unknown type {ty}")


def reset() -> None:
    global graph_breaker
    graph_breaker = None
    global random_state
    random_state = None