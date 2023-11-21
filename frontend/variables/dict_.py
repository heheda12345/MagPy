from typing import TYPE_CHECKING, Optional, Any, Callable
from .base import Variable, HelperFunctions
from ..fx_graph import NodeArgs, FxGraph
from ..store_pos import StorePos, StoreInIndex
from .tensor import TensorVar
import torch
if TYPE_CHECKING:
    from ..pycode_generator import GraphFnCodegen, GuardFnCodegen
    from ..object_table import ObjectTable


class DictVar(Variable):
    vars: list[Variable]
    obj_ids: list[int]
    length: int

    def __init__(self,
                 value: dict[Any, Any],
                 need_guard_check: bool,
                 helper_functions: HelperFunctions,
                 fx_graph: Optional[FxGraph] = None,
                 extract_code_at_start: list[StorePos] = []) -> None:
        super().__init__(need_guard_check, value, extract_code_at_start)
        self.value = value
        self.length = len(value)
        self.vars = []
        self.obj_ids = []
        for key, obj in self.value.items():
            assert not isinstance(key, torch.Tensor)
            if isinstance(key, str):
                new_extract: list[StorePos] = [
                    StoreInIndex(pos, id(obj), f"'{key}'")
                    for pos in self.extract_code_at_start
                ]
            else:
                new_extract = [
                    StoreInIndex(pos, id(obj), str(key))
                    for pos in self.extract_code_at_start
                ]
            var = helper_functions.get_or_make_var(obj, need_guard_check,
                                                   fx_graph, new_extract)
            self.vars.append(var)
            self.obj_ids.append(id(obj))

    def make_guard_inner(self, codegen: "GuardFnCodegen",
                         pos: StorePos) -> None:
        codegen.add_check(f"isinstance({pos}, dict)")
        codegen.add_check(f"len({pos}) == {self.length}")
        for key, obj in zip(self.value.keys(), self.vars):
            if not isinstance(obj, TensorVar):
                if isinstance(key, str):
                    obj.make_guard_inner(
                        codegen, StoreInIndex(pos, id(obj), f"'{key}'"))
                else:
                    obj.make_guard_inner(codegen,
                                         StoreInIndex(pos, id(obj), str(key)))

    def make_output_inner(self, name_in_graph_fn: str, store_pos: StorePos,
                          codegen: "GraphFnCodegen", in_return: bool,
                          idx: int) -> None:
        oldest = self.get_oldest_var()
        for j, (idx_j, var) in enumerate(zip(self.obj_ids, self.vars)):
            var.make_output(f"{name_in_graph_fn}_{j}", store_pos, codegen,
                            False, idx_j)

        if len(oldest.extract_code_at_start) > 0:
            assert isinstance(oldest, DictVar)
            old_store_pos = oldest.extract_code_at_start[0]
            codegen.add_stmt(f"{old_store_pos}.clear()")
            for i, key in enumerate(self.value.keys()):
                codegen.add_stmt(
                    f"{old_store_pos}[{key}]={name_in_graph_fn}_{i}")
            codegen.output(name_in_graph_fn, store_pos, str(old_store_pos),
                           in_return, idx)
        else:
            items = []
            for key, j in zip(self.value.keys(), range(len(self.vars))):
                if isinstance(key, str):
                    key_part = f"'{key}'"
                else:
                    key_part = key
                item = f'{key_part}: {name_in_graph_fn}_{j}'
                items.append(item)
            target = f"{{{', '.join(i for i in items)}}}"
            codegen.output(name_in_graph_fn, store_pos,
                           target if len(self.vars) > 0 else "{}", in_return,
                           idx)

    @classmethod
    def from_value(cls,
                   value: dict[Any, Any],
                   need_guard_check: bool,
                   helper_functions: HelperFunctions,
                   fx_graph: Optional[FxGraph] = None,
                   extract_code_at_start: list[StorePos] = []) -> "DictVar":
        return cls(value, need_guard_check, helper_functions, fx_graph,
                   extract_code_at_start)

    def as_fx_node(self) -> NodeArgs:
        return self.value

    def add_subvars_to_table(self, table: 'ObjectTable') -> None:
        for key, var, idx in zip(self.value.keys(), self.vars, self.obj_ids):
            old_var = table.get_or_none_by_id(idx)
            if old_var is not None:
                if isinstance(key, str):
                    new_extract: list[StorePos] = [
                        StoreInIndex(pos, idx, f"'{key}'")
                        for pos in self.extract_code_at_start
                    ]
                else:
                    new_extract = [
                        StoreInIndex(pos, idx, str(key))
                        for pos in self.extract_code_at_start
                    ]
                old_var.extract_code_at_start.extend(new_extract)
                old_var.need_guard_check |= self.need_guard_check
            else:
                table.add_by_id(var, idx)
                var.add_subvars_to_table(table)


class OrderedDictVar(DictVar):

    def __init__(self,
                 value: dict[Any, Any],
                 need_guard_check: bool,
                 helper_functions: HelperFunctions,
                 fx_graph: Optional[FxGraph] = None,
                 extract_code_at_start: list[StorePos] = []) -> None:
        super().__init__(value, need_guard_check, helper_functions, fx_graph,
                         extract_code_at_start)

    @classmethod
    def from_value(
            cls,
            value: dict[Any, Any],
            need_guard_check: bool,
            helper_functions: HelperFunctions,
            fx_graph: Optional[FxGraph] = None,
            extract_code_at_start: list[StorePos] = []) -> "OrderedDictVar":
        return cls(value, need_guard_check, helper_functions, fx_graph,
                   extract_code_at_start)

    def make_guard_inner(self, codegen: "GuardFnCodegen",
                         pos: StorePos) -> None:
        codegen.add_check(f"isinstance({pos}, dict)")
        codegen.add_check(f"len({pos}) == {self.length}")
        for key, var in zip(self.value.keys(), self.vars):
            if not isinstance(var, TensorVar):
                if isinstance(key, str):
                    var.make_guard_inner(
                        codegen, StoreInIndex(pos, id(var), f"'{key}'"))
                else:
                    var.make_guard_inner(codegen,
                                         StoreInIndex(pos, id(var), str(key)))
        # TODO: check order

    def make_output_inner(self, name_in_graph_fn: str, store_pos: StorePos,
                          codegen: "GraphFnCodegen", in_return: bool,
                          idx: int) -> None:
        oldest = self.get_oldest_var()
        for j, (idx_j, var) in enumerate(zip(self.obj_ids, self.vars)):
            var.make_output(f"{name_in_graph_fn}_{j}", store_pos, codegen,
                            False, idx_j)
        if len(oldest.extract_code_at_start) > 0:
            assert isinstance(oldest, DictVar)
            old_store_pos = oldest.extract_code_at_start[0]
            codegen.add_stmt(f"{old_store_pos}.clear()")
            for i, key in enumerate(self.value.keys()):
                codegen.add_stmt(
                    f"{old_store_pos}[{key}]={name_in_graph_fn}_{i}")
            codegen.output(name_in_graph_fn, store_pos, str(old_store_pos),
                           in_return, idx)
        else:
            codegen.add_import("collections")

            def to_str(value: Any) -> str:
                if isinstance(value, str):
                    return f"'{value}'"
                else:
                    return str(value)

            codegen.output(
                name_in_graph_fn, store_pos,
                f"collections.OrderedDict([{','.join(f'({to_str(key)}, {name_in_graph_fn}_{j})' for key, j in zip(self.value.keys(), range(len(self.vars))))}])"
                if len(self.vars) > 0 else "{}", in_return, idx)
