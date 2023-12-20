from typing import TYPE_CHECKING, Optional, Tuple, Any, Callable
from types import CellType
from .base import Variable, HelperFunctions
from ..fx_graph import NodeArgs, FxGraph
from ..store_pos import StorePos, StoreInAttr, StoreInFreeVar
from ..c_api import parse_mapproxyobject, parse_cell
import torch
if TYPE_CHECKING:
    from ..pycode_generator import GraphFnCodegen, GuardFnCodegen
    from ..object_table import ObjectTable


class CellVar(Variable):
    sub_var: Variable
    sub_id: int

    def __init__(self, value: CellType, need_guard_check: bool,
                 helper_functions: HelperFunctions, fx_graph: Optional[FxGraph],
                 extract_code_at_start: list[StorePos]) -> None:
        super().__init__(need_guard_check, value, extract_code_at_start)
        assert len(extract_code_at_start) > 0
        sub_obj = parse_cell(value)
        new_extract: list[StorePos] = [
            StoreInAttr(pos, id(value), "cell_contents")
            for pos in self.extract_code_at_start
        ]
        self.sub_var = helper_functions.get_or_make_var(sub_obj,
                                                        need_guard_check,
                                                        fx_graph, new_extract)
        self.sub_id = id(sub_obj)

    def make_guard_inner(self, codegen: "GuardFnCodegen",
                         pos: StorePos) -> None:
        codegen.add_import_from("types", "CellType")
        codegen.add_check(f"isinstance({pos}, CellType)")
        self.sub_var.make_guard_inner(
            codegen, StoreInAttr(pos, self.sub_id, "cell_contents"))

    def make_output_inner(self, name_in_graph_fn: str, store_pos: StorePos,
                          codegen: "GraphFnCodegen", in_return: bool,
                          idx: int) -> None:
        codegen.add_import("inspect")
        codegen.add_import_from("frontend.c_api", "get_from_freevars")
        codegen.output(f"{name_in_graph_fn}_frame", store_pos,
                       "inspect.currentframe().f_back", False, idx)
        extract_pos = self.fetch_extract_code_at_start()
        assert len(extract_pos) > 0
        extract_pos[0].add_name_to_fn(codegen)
        cell_pos = extract_pos[0]
        if isinstance(cell_pos, StoreInFreeVar):
            codegen.output(
                name_in_graph_fn, store_pos,
                f"get_from_freevars({name_in_graph_fn}_frame, {cell_pos.free_idx})",
                False, idx)
        else:
            raise NotImplementedError

    @classmethod
    def from_value(cls, value: CellType, need_guard_check: bool,
                   helper_functions: HelperFunctions,
                   fx_graph: Optional[FxGraph],
                   extract_code_at_start: list[StorePos]) -> "CellVar":
        return cls(value, need_guard_check, helper_functions, fx_graph,
                   extract_code_at_start)

    def add_subvars_to_table(self, table: 'ObjectTable') -> None:
        old_var = table.get_or_none_by_id(self.sub_id)
        if old_var is not None:
            new_extract: list[StorePos] = [
                StoreInAttr(pos, self.sub_id, "cell_contents")
                for pos in self.extract_code_at_start
            ]
            old_var.extract_code_at_start.extend(new_extract)
            old_var.need_guard_check |= self.need_guard_check
        else:
            table.add_by_id(self.sub_var, self.sub_id)
            self.sub_var.add_subvars_to_table(table)


class MappingProxyVar(Variable):
    sub_var: Variable
    sub_id: int

    def __init__(self, value: Any, need_guard_check: bool,
                 helper_functions: HelperFunctions, fx_graph: Optional[FxGraph],
                 extract_code_at_start: list[StorePos]) -> None:
        super().__init__(need_guard_check, value, extract_code_at_start)
        # assert len(extract_code_at_start) > 0
        sub_obj = parse_mapproxyobject(value)
        new_extract: list[StorePos] = []
        self.sub_var = helper_functions.get_or_make_var(sub_obj,
                                                        need_guard_check,
                                                        fx_graph, new_extract)
        self.sub_id = id(sub_obj)

    @classmethod
    def from_value(cls, value: Any, need_guard_check: bool,
                   helper_functions: HelperFunctions,
                   fx_graph: Optional[FxGraph],
                   extract_code_at_start: list[StorePos]) -> "MappingProxyVar":
        return cls(value, need_guard_check, helper_functions, fx_graph,
                   extract_code_at_start)

    def make_guard_inner(self, codegen: 'GuardFnCodegen',
                         pos: StorePos) -> None:
        raise ValueError("TOOD")

    def make_output_inner(self, name_in_graph_fn: str, store_pos: StorePos,
                          codegen: 'GraphFnCodegen', in_return: bool,
                          idx: int) -> None:
        raise ValueError("TOOD")

    def add_subvars_to_table(self, table: 'ObjectTable') -> None:
        old_var = table.get_or_none_by_id(self.sub_id)
        if old_var is not None:
            # TODO: handle extract_code_at_start
            pass
        else:
            table.add_by_id(self.sub_var, self.sub_id)
            self.sub_var.add_subvars_to_table(table)
