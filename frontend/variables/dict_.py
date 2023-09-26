from typing import TYPE_CHECKING, Optional, Tuple, Any
from .base import Variable
from ..fx_graph import NodeArgs, FxGraph
from ..store_pos import StorePos, StoreInIndex
from .tensor import TensorVar
import torch
if TYPE_CHECKING:
    from ..pycode_generator import GraphFnCodegen, GuardFnCodegen
    from ..object_table import ReadOnlyObjectTable, ObjectTable


class DictVar(Variable):
    vars: list[Variable]
    obj_ids: list[int]
    length: int

    def __init__(self,
                 value: dict[Any, Any],
                 need_guard_check: bool,
                 object_table: 'ReadOnlyObjectTable',
                 fx_graph: Optional[FxGraph] = None,
                 extract_code_at_start: list[StorePos] = []) -> None:
        super().__init__(need_guard_check, extract_code_at_start)
        self.value = value
        self.length = len(value)
        self.vars = []
        self.obj_ids = []
        from . import make_var_from_value
        for key, value in self.value.items():
            assert not isinstance(key, torch.Tensor)
            new_extract: list[StorePos] = [
                StoreInIndex(pos, str(key))
                for pos in self.extract_code_at_start
            ]
            var = object_table.get_or_make_var(value, need_guard_check,
                                               fx_graph, new_extract)
            self.vars.append(var)
            self.obj_ids.append(id(value))

    def make_guard_inner(self, codegen: "GuardFnCodegen",
                         pos: StorePos) -> None:
        codegen.add_check(f"len({pos}) == {self.length}")
        for key, obj in zip(self.value.keys(), self.vars):
            if not isinstance(obj, TensorVar):
                obj.make_guard_inner(codegen, StoreInIndex(pos, str(key)))

    def make_output(self, name_in_graph_fn: str, store_pos: StorePos,
                    codegen: "GraphFnCodegen") -> None:
        for j, var in enumerate(self.vars):
            var.make_temp(f"{name_in_graph_fn}_{j}", store_pos, codegen)

        codegen.output(
            name_in_graph_fn, store_pos,
            f"{{{','.join(f'{key}: {name_in_graph_fn}_{j}' for key, j in zip(self.value.keys(), range(len(self.vars))))}}}"
        )

    def make_temp(self, name_in_graph_fn: str, store_pos: StorePos,
                  codegen: "GraphFnCodegen") -> None:
        for j, var in enumerate(self.vars):
            var.make_temp(f"{name_in_graph_fn}_{j}", store_pos, codegen)
        codegen.add_temp(
            name_in_graph_fn, store_pos,
            f"({','.join(f'{name_in_graph_fn}_{j}' for j in range(len(self.vars)))})"
        )

    @classmethod
    def from_value(cls,
                   value: dict[Any, Any],
                   need_guard_check: bool,
                   object_table: 'ReadOnlyObjectTable',
                   fx_graph: Optional[FxGraph] = None,
                   extract_code_at_start: list[StorePos] = []) -> "DictVar":
        return cls(value, need_guard_check, object_table, fx_graph,
                   extract_code_at_start)

    def as_fx_node(self) -> NodeArgs:
        return self.value

    def add_subvars_to_table(self, table: 'ObjectTable') -> None:
        for i, (var, idx) in enumerate(zip(self.vars, self.obj_ids)):
            old_var = table.get_or_none(idx)
            if old_var is not None:
                new_extract: list[StorePos] = [
                    StoreInIndex(pos, i) for pos in self.extract_code_at_start
                ]
                old_var.extract_code_at_start.extend(new_extract)
                old_var.need_guard_check |= self.need_guard_check
            else:
                table.add_by_id(var, idx)
                var.add_subvars_to_table(table)