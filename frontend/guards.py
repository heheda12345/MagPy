import dataclasses
import typing
from typing import Any


@dataclasses.dataclass
class GuardedCode:
    check_fn: typing.Callable[..., Any]
    graph_fn: typing.Callable[..., Any]
