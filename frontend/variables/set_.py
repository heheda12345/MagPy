from typing import TYPE_CHECKING, Optional, Tuple, Any
from .base import Variable
from ..fx_graph import NodeArgs, FxGraph
from ..store_pos import StorePos, StoreInIndex
import torch
if TYPE_CHECKING:
    from ..pycode_generator import GraphFnCodegen, GuardFnCodegen
    from ..object_table import ReadOnlyObjectTable, ObjectTable


class SetVar(Variable):
    vars: list[Variable]
    obj_ids: list[int]
    length: int

    def __init__(self,
                 value: set[Any],
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
                StoreInIndex(pos, i, False)
                for pos in self.extract_code_at_start
            ]
            var = object_table.get_or_make_var(obj, need_guard_check, fx_graph,
                                               new_extract)
            self.vars.append(var)
            self.obj_ids.append(id(obj))

    def make_guard_inner(self, codegen: "GuardFnCodegen",
                         pos: StorePos) -> None:
        codegen.add_check(f"len({pos}) == {self.length}")

    def make_output(self, name_in_graph_fn: str, store_pos: StorePos,
                    codegen: "GraphFnCodegen") -> None:
        for j, var in enumerate(self.vars):
            var.make_temp(f"{name_in_graph_fn}_{j}", store_pos, codegen)

        codegen.output(
            name_in_graph_fn, store_pos,
            f"{{{','.join(f'{name_in_graph_fn}_{j}' for j in range(len(self.vars)))},}}"
        )

    def make_temp(self, name_in_graph_fn: str, store_pos: StorePos,
                  codegen: "GraphFnCodegen") -> None:
        codegen.add_temp(name_in_graph_fn, store_pos, str(self.value))

    @classmethod
    def from_value(cls,
                   value: set[Any],
                   need_guard_check: bool,
                   object_table: 'ReadOnlyObjectTable',
                   fx_graph: Optional[FxGraph] = None,
                   extract_code_at_start: list[StorePos] = []) -> "SetVar":
        return cls(value, need_guard_check, object_table, fx_graph,
                   extract_code_at_start)

    def as_fx_node(self) -> NodeArgs:
        return self.value

    def add_subvars_to_table(self, table: 'ObjectTable') -> None:
        pass