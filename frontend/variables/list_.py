from typing import TYPE_CHECKING, Optional, Any
from copy import copy
from .base import Variable
from ..fx_graph import NodeArgs, FxGraph
from ..store_pos import StorePos, StoreInIndex
if TYPE_CHECKING:
    from ..pycode_generator import GraphFnCodegen, GuardFnCodegen
    from ..object_table import ReadOnlyObjectTable, ObjectTable


class ListVar(Variable):
    vars: list[Variable]
    obj_ids: list[int]
    length: int

    def __init__(self,
                 value: list[Any],
                 need_guard_check: bool,
                 object_table: 'ReadOnlyObjectTable',
                 fx_graph: Optional[FxGraph] = None,
                 extract_code_at_start: list[StorePos] = []) -> None:
        super().__init__(need_guard_check, extract_code_at_start)
        self.value = value
        self.length = len(value)
        self.vars = []
        self.obj_ids = []
        for i, obj in enumerate(value):
            new_extract: list[StorePos] = [
                StoreInIndex(pos, i) for pos in self.extract_code_at_start
            ]
            var = object_table.get_or_make_var(obj, need_guard_check, fx_graph,
                                               new_extract)
            self.vars.append(var)
            self.obj_ids.append(id(obj))

    def make_guard_inner(self, codegen: "GuardFnCodegen",
                         pos: StorePos) -> None:
        codegen.add_check(f"isinstance({pos}, list)")
        codegen.add_check(f"len({pos}) == {self.length}")
        for i, obj in enumerate(self.vars):
            obj.make_guard_inner(codegen, StoreInIndex(pos, i))

    def make_output_inner(self, name_in_graph_fn: str, store_pos: StorePos,
                          codegen: "GraphFnCodegen", in_return: bool,
                          idx: int) -> None:
        oldest = self.get_oldest_var()
        for j, (idx_j, var) in enumerate(zip(self.obj_ids, self.vars)):
            var.make_output(f"{name_in_graph_fn}_{j}", store_pos, codegen,
                            False, idx_j)
        if len(oldest.extract_code_at_start) > 0:
            assert isinstance(oldest, ListVar)
            old_store_pos = oldest.extract_code_at_start[0]
            codegen.add_stmt(f"{old_store_pos}.clear()")
            for i in range(self.length):
                codegen.add_stmt(
                    f"{old_store_pos}.append({name_in_graph_fn}_{i})")
            codegen.output(name_in_graph_fn, store_pos, str(old_store_pos),
                           in_return, idx)
        else:
            codegen.output(
                name_in_graph_fn, store_pos,
                f"[{','.join(f'{name_in_graph_fn}_{j}' for j in range(len(self.vars)))},]"
                if len(self.vars) > 0 else "[]", in_return, idx)

    @classmethod
    def from_value(cls,
                   value: list[Any],
                   need_guard_check: bool,
                   object_table: 'ReadOnlyObjectTable',
                   fx_graph: Optional[FxGraph] = None,
                   extract_code_at_start: list[StorePos] = []) -> "ListVar":
        return cls(value, need_guard_check, object_table, fx_graph,
                   extract_code_at_start)

    def as_fx_node(self) -> NodeArgs:
        return self.value

    def add_subvars_to_table(self, table: 'ObjectTable') -> None:
        for i, (var, idx) in enumerate(zip(self.vars, self.obj_ids)):
            old_var = table.get_or_none_by_id(idx)
            if old_var is not None:
                new_extract: list[StorePos] = [
                    StoreInIndex(pos, i) for pos in self.extract_code_at_start
                ]
                old_var.extract_code_at_start.extend(new_extract)
                old_var.need_guard_check |= self.need_guard_check
            else:
                table.add_by_id(var, idx)
                var.add_subvars_to_table(table)