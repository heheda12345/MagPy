import dataclasses
import typing


@dataclasses.dataclass
class GuardedCode:
    check_fn: typing.Callable
    graph_fn: typing.Callable
