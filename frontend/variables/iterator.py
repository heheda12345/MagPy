from typing import TYPE_CHECKING, Optional, Tuple, Any, Callable, Iterable
from .base import Variable, HelperFunctions
from ..fx_graph import NodeArgs, FxGraph
from ..store_pos import StorePos, StoreInIndex
import torch
from ..c_api import parse_rangeiterobject
if TYPE_CHECKING:
    from ..pycode_generator import GraphFnCodegen, GuardFnCodegen
    from ..object_table import ObjectTable


class IteratorVar(Variable):
    parent_var: Optional[Variable]
    parent_idx: int
    num_iters: int

    def __init__(
        self,
        value: Any,
        parent_var: Optional[Variable],
        parent_idx: int,
        num_iters: int,
        need_guard_check: bool,
        extract_code_at_start: list[StorePos],
    ) -> None:
        super().__init__(need_guard_check, value, extract_code_at_start)
        assert not need_guard_check
        self.parent_var = parent_var
        self.parent_idx = parent_idx
        self.num_iters = num_iters

    @classmethod
    def from_parent_var(cls, value: Any, parent_var: Optional[Variable],
                        parent_idx: int, num_iters: int, need_guard_check: bool,
                        extract_code_at_start: list[StorePos]) -> "IteratorVar":
        return cls(value, parent_var, parent_idx, num_iters, need_guard_check,
                   extract_code_at_start)

    def make_guard_inner(self, codegen: "GuardFnCodegen",
                         store_pos: StorePos) -> None:
        raise NotImplementedError()

    def make_output_inner(self, name_in_graph_fn: str, store_pos: StorePos,
                          codegen: "GraphFnCodegen", in_return: bool,
                          idx: int) -> None:
        if self.parent_var is None:
            raise ValueError("cannot gen output for None parent_var")
        self.parent_var.make_output(f"{name_in_graph_fn}_iterable", store_pos,
                                    codegen, False, idx)
        codegen.output(name_in_graph_fn, store_pos,
                       f"{name_in_graph_fn}_iterable.__iter__()", in_return,
                       idx)
        for i in range(self.num_iters):
            codegen.add_stmt(f"{name_in_graph_fn}.__next__()")


class RangeIterVar(Variable):
    index: int
    start: int
    step: int
    len: int

    def __init__(self, obj: Any, need_guard_check: bool,
                 extract_code_at_start: list[StorePos]) -> None:
        super().__init__(need_guard_check, obj, extract_code_at_start)
        self.index, self.start, self.step, self.len = parse_rangeiterobject(obj)

    def make_guard_inner(self, codegen: "GuardFnCodegen",
                         pos: StorePos) -> None:
        codegen.add_import("frontend.c_api")
        codegen.add_check(
            f"frontend.c_api.parse_rangeiterobject({pos}) == ({self.index}, {self.start}, {self.step}, {self.len})"
        )

    def make_output_inner(self, name_in_graph_fn: str, store_pos: StorePos,
                          codegen: "GraphFnCodegen", in_return: bool,
                          idx: int) -> None:
        codegen.add_import("frontend.c_api")
        codegen.output(
            name_in_graph_fn, store_pos,
            f"frontend.c_api.make_rangeiterobject({self.index}, {self.start}, {self.step}, {self.len})",
            in_return, idx)

    @classmethod
    def from_value(cls, value: Any, need_guard_check: bool,
                   _helper_functions: HelperFunctions,
                   _fx_graph: Optional[FxGraph],
                   extract_code_at_start: list[StorePos]) -> "RangeIterVar":
        return cls(value, need_guard_check, extract_code_at_start)
