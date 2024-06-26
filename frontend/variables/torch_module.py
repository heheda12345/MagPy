from typing import TYPE_CHECKING, Union, Optional, Callable, Any
import torch
import torch.fx
from frontend.pycode_generator import GraphFnCodegen, GuardFnCodegen
from .base import Variable, HelperFunctions
from ..utils import ScalarType
from ..fx_graph import FxGraph, NodeArgs
from ..store_pos import StorePos, StoreInIndex
if TYPE_CHECKING:
    from ..pycode_generator import GraphFnCodegen, GuardFnCodegen
    from ..object_table import ObjectTable


class TorchModuleVar(Variable):

    def __init__(self, value: torch.nn.Module, need_guard_check: bool,
                 extract_code_at_start: list[StorePos]) -> None:
        super().__init__(need_guard_check, value, extract_code_at_start)

    @classmethod
    def from_value(
        cls,
        value: torch.nn.Module,
        need_guard_check: bool,
        helper_functions: HelperFunctions,
        _fx_graph: Optional[FxGraph],
        extract_code_at_start: list[StorePos],
    ) -> "TorchModuleVar":
        if isinstance(value, torch.nn.Sequential):
            return TorchSequentialVar(value, need_guard_check, helper_functions,
                                      extract_code_at_start)
        elif isinstance(value, torch.nn.ModuleList):
            return TorchModuleListVar(value, need_guard_check, helper_functions,
                                      extract_code_at_start)
        else:
            return cls(value, need_guard_check, extract_code_at_start)

    def make_guard_inner(self, codegen: GuardFnCodegen, pos: StorePos) -> None:
        codegen.add_id_check((f"id({pos}) == {id(self.obj)}", pos), self.obj)

    def make_output_inner(self, name_in_graph_fn: str, store_pos: StorePos,
                          codegen: "GraphFnCodegen", in_return: bool,
                          idx: int) -> None:
        extract_pos = self.fetch_extract_code_at_start()
        assert len(extract_pos) > 0
        extract_pos[0].add_name_to_fn(codegen)
        codegen.output(name_in_graph_fn, store_pos, str(extract_pos[0]),
                       in_return, idx)

    def as_fx_node(self) -> NodeArgs:
        return self.obj
        raise ValueError("Cannot convert a module to a node")


class TorchSequentialVar(TorchModuleVar):
    submodules: list[TorchModuleVar]
    submodule_ids: list[int]

    def __init__(self, value: torch.nn.Sequential, need_guard_check: bool,
                 helper_functions: HelperFunctions,
                 extract_code_at_start: list[StorePos]) -> None:
        super().__init__(value, need_guard_check, extract_code_at_start)
        self.submodules = []
        self.submodule_ids = []
        for i, m in enumerate(value):
            new_extract: list[StorePos] = [
                StoreInIndex(pos, id(m), i)
                for pos in self.extract_code_at_start
            ]
            var = helper_functions.get_or_make_var(m, need_guard_check, None,
                                                   new_extract)
            assert isinstance(var, TorchModuleVar)
            self.submodules.append(var)
            self.submodule_ids.append(id(m))

    def add_subvars_to_table(self, table: 'ObjectTable') -> None:
        for i, (var, idx) in enumerate(zip(self.submodules,
                                           self.submodule_ids)):
            old_var = table.get_or_none_by_id(idx)
            if old_var is not None:
                new_extract: list[StorePos] = [
                    StoreInIndex(pos, idx, i)
                    for pos in self.extract_code_at_start
                ]
                old_var.extract_code_at_start.extend(new_extract)
                old_var.need_guard_check |= self.need_guard_check
            else:
                table.add_by_id(var, idx)
                var.add_subvars_to_table(table)


class TorchModuleListVar(TorchModuleVar):
    submodules: list[TorchModuleVar]
    submodule_ids: list[int]

    def __init__(self, value: torch.nn.ModuleList, need_guard_check: bool,
                 helper_functions: HelperFunctions,
                 extract_code_at_start: list[StorePos]) -> None:
        super().__init__(value, need_guard_check, extract_code_at_start)
        self.submodules = []
        self.submodule_ids = []
        for i, m in enumerate(value):
            new_extract: list[StorePos] = [
                StoreInIndex(pos, id(m), i)
                for pos in self.extract_code_at_start
            ]
            var = helper_functions.get_or_make_var(m, need_guard_check, None,
                                                   new_extract)
            assert isinstance(var, TorchModuleVar)
            self.submodules.append(var)
            self.submodule_ids.append(id(m))

    def add_subvars_to_table(self, table: 'ObjectTable') -> None:
        for i, (var, idx) in enumerate(zip(self.submodules,
                                           self.submodule_ids)):
            old_var = table.get_or_none_by_id(idx)
            if old_var is not None:
                new_extract: list[StorePos] = [
                    StoreInIndex(pos, idx, i)
                    for pos in self.extract_code_at_start
                ]
                old_var.extract_code_at_start.extend(new_extract)
                old_var.need_guard_check |= self.need_guard_check
            else:
                table.add_by_id(var, idx)
                var.add_subvars_to_table(table)