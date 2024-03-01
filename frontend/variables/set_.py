from typing import TYPE_CHECKING, Optional, Callable, Any
from .base import Variable, HelperFunctions
from ..fx_graph import NodeArgs, FxGraph
from ..store_pos import StorePos, StoreInIndex
import torch
if TYPE_CHECKING:
    from ..pycode_generator import GraphFnCodegen, GuardFnCodegen
    from ..object_table import ObjectTable


class SetVar(Variable):
    vars: list[Variable]
    obj_ids: list[int]
    length: int

    def __init__(self, value: set[Any], need_guard_check: bool,
                 helper_functions: HelperFunctions, fx_graph: Optional[FxGraph],
                 extract_code_at_start: list[StorePos]) -> None:
        super().__init__(need_guard_check, value, extract_code_at_start)
        self.value = value
        self.length = len(value)
        self.vars = []
        self.obj_ids = []
        for i, obj in enumerate(value):
            new_extract: list[StorePos] = [
                StoreInIndex(pos, id(obj), i, False)
                for pos in self.extract_code_at_start
            ]
            var = helper_functions.get_or_make_var(obj, need_guard_check,
                                                   fx_graph, new_extract)
            self.vars.append(var)
            self.obj_ids.append(id(obj))

    def make_guard_inner(self, codegen: "GuardFnCodegen",
                         pos: StorePos) -> None:
        codegen.add_check((f'isinstance({pos}, set)', pos))
        codegen.add_check((f"len({pos}) == {self.length}", pos))
        for i, (var, obj) in enumerate(zip(self.vars, self.obj_ids)):
            var.make_guard_inner(codegen, StoreInIndex(pos, obj, i, False))

    def make_output_inner(self, name_in_graph_fn: str, store_pos: StorePos,
                          codegen: "GraphFnCodegen", in_return: bool,
                          idx: int) -> None:
        for j, (idx_j, var) in enumerate(zip(self.obj_ids, self.vars)):
            var.make_output(f"{name_in_graph_fn}_{j}", store_pos, codegen,
                            False, idx_j)

        codegen.output(
            name_in_graph_fn, store_pos,
            f"{{{','.join(f'{name_in_graph_fn}_{j}' for j in range(len(self.vars)))},}}"
            if len(self.vars) > 0 else "set()", in_return, idx)

    @classmethod
    def from_value(cls, value: set[Any], need_guard_check: bool,
                   helper_functions: HelperFunctions,
                   fx_graph: Optional[FxGraph],
                   extract_code_at_start: list[StorePos]) -> "SetVar":
        return cls(value, need_guard_check, helper_functions, fx_graph,
                   extract_code_at_start)

    def as_fx_node(self) -> NodeArgs:
        return self.value

    def add_subvars_to_table(self, table: 'ObjectTable') -> None:
        for i, (var, idx) in enumerate(zip(self.vars, self.obj_ids)):
            old_var = table.get_or_none_by_id(idx)
            if old_var is not None:
                new_extract: list[StorePos] = [
                    StoreInIndex(pos, idx, i, False)
                    for pos in self.extract_code_at_start
                ]
                old_var.extract_code_at_start.extend(new_extract)
                old_var.need_guard_check |= self.need_guard_check
            else:
                table.add_by_id(var, idx)
                var.add_subvars_to_table(table)


class FrozensetVar(Variable):
    vars: list[Variable]
    obj_ids: list[int]
    length: int

    def __init__(self, value: frozenset[Any], need_guard_check: bool,
                 helper_functions: HelperFunctions, fx_graph: Optional[FxGraph],
                 extract_code_at_start: list[StorePos]) -> None:
        super().__init__(need_guard_check, value, extract_code_at_start)
        self.value = value
        self.length = len(value)
        self.vars = []
        self.obj_ids = []
        for i, obj in enumerate(value):
            new_extract: list[StorePos] = [
                StoreInIndex(pos, id(obj), i, False)
                for pos in self.extract_code_at_start
            ]
            var = helper_functions.get_or_make_var(obj, need_guard_check,
                                                   fx_graph, new_extract)
            self.vars.append(var)
            self.obj_ids.append(id(obj))

    def make_guard_inner(self, codegen: "GuardFnCodegen",
                         pos: StorePos) -> None:
        codegen.add_check((f'isinstance({pos}, frozenset)', pos))
        codegen.add_check((f"len({pos}) == {self.length}", pos))
        for i, (var, obj) in enumerate(zip(self.vars, self.obj_ids)):
            var.make_guard_inner(codegen, StoreInIndex(pos, obj, i, False))

    def make_output_inner(self, name_in_graph_fn: str, store_pos: StorePos,
                          codegen: "GraphFnCodegen", in_return: bool,
                          idx: int) -> None:
        for j, (idx_j, var) in enumerate(zip(self.obj_ids, self.vars)):
            var.make_output(f"{name_in_graph_fn}_{j}", store_pos, codegen,
                            False, idx_j)

        codegen.output(
            name_in_graph_fn, store_pos,
            f"{{{','.join(f'{name_in_graph_fn}_{j}' for j in range(len(self.vars)))},}}"
            if len(self.vars) > 0 else "frozenset()", in_return, idx)

    @classmethod
    def from_value(cls, value: frozenset[Any], need_guard_check: bool,
                   helper_functions: HelperFunctions,
                   fx_graph: Optional[FxGraph],
                   extract_code_at_start: list[StorePos]) -> "FrozensetVar":
        return cls(value, need_guard_check, helper_functions, fx_graph,
                   extract_code_at_start)

    def as_fx_node(self) -> NodeArgs:
        return self.value

    def add_subvars_to_table(self, table: 'ObjectTable') -> None:
        for i, (var, idx) in enumerate(zip(self.vars, self.obj_ids)):
            old_var = table.get_or_none_by_id(idx)
            if old_var is not None:
                new_extract: list[StorePos] = [
                    StoreInIndex(pos, idx, i, False)
                    for pos in self.extract_code_at_start
                ]
                old_var.extract_code_at_start.extend(new_extract)
                old_var.need_guard_check |= self.need_guard_check
            else:
                table.add_by_id(var, idx)
                var.add_subvars_to_table(table)