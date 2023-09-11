from dataclasses import dataclass
from abc import abstractmethod
from typing import Any, TYPE_CHECKING, Optional
from ..fx_graph import FxGraph
from ..cache import StorePos
if TYPE_CHECKING:
    import torch.fx
    from ..pycode_generator import GraphFnCodegen, GuardFnCodegen
    from ..fx_graph import FxGraph, ProxyArgs


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
                   fx_graph: Optional[FxGraph] = None,
                   extract_code_at_start: str = "") -> 'Variable':
        raise NotImplementedError

    def make_guard(self, codegen: "GuardFnCodegen") -> None:
        if self.need_guard_check:
            self.make_guard_inner(codegen)

    @abstractmethod
    def make_guard_inner(self, codegen: "GuardFnCodegen") -> None:
        raise NotImplementedError

    @abstractmethod
    def make_output(self, name_in_graph_fn: str, store_pos: StorePos,
                    codegen: "GraphFnCodegen") -> None:
        raise NotImplementedError

    @abstractmethod
    def as_proxy(self) -> "ProxyArgs":
        raise NotImplementedError
