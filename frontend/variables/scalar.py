from typing import TYPE_CHECKING, Union, Optional, Callable, Any

import torch.fx
import numpy as np
from .. import config
from .base import Variable, HelperFunctions
from ..pycode_writer import get_float_string
from ..fx_graph import NodeArgs, FxGraph
from ..store_pos import StorePos
from ..pycode_writer import new_name
from ..utils import ScalarType
from .. import dynamic as dyn
if TYPE_CHECKING:
    from ..pycode_generator import GraphFnCodegen, GuardFnCodegen


class ScalarVar(Variable):
    value_fix: bool
    fx_node: Optional[torch.fx.Node]

    def __init__(self, value: ScalarType, value_fix: bool,
                 need_guard_check: bool, fx_node: Optional[torch.fx.Node],
                 extract_code_at_start: list[StorePos]) -> None:
        super().__init__(need_guard_check, value, extract_code_at_start)
        if isinstance(value, bool) and not value_fix:
            raise NotImplementedError
        if not value_fix:
            assert fx_node is not None
        self.value_fix = value_fix
        self.fx_node = fx_node

    def make_guard_inner(self, codegen: "GuardFnCodegen",
                         pos: StorePos) -> None:
        codegen.add_check(f"isinstance({pos}, {type(self.obj).__name__})")
        if self.value_fix:
            if type(self.obj) == float:
                codegen.add_check(f"{pos} == {get_float_string(self.obj)}")
                codegen.add_import("struct")
            elif isinstance(self.obj, str):
                codegen.add_check(f"{pos} == '{self.obj}'")
            else:
                codegen.add_check(f"{pos} == {self.obj}")

    def make_output_inner(self, name_in_graph_fn: str, store_pos: StorePos,
                          codegen: "GraphFnCodegen", in_return: bool,
                          idx: int) -> None:
        if self.value_fix:
            if type(self.obj) == float:
                codegen.output(name_in_graph_fn, store_pos,
                               f"{get_float_string(self.obj)} # {self.obj}",
                               in_return, idx)
                codegen.add_import("struct")
            elif isinstance(self.obj, str):
                codegen.output(name_in_graph_fn, store_pos, f"'{self.obj}'",
                               in_return, idx)
            else:
                codegen.output(name_in_graph_fn, store_pos, str(self.obj),
                               in_return, idx)
        else:
            name_in_graph_output = codegen.add_graph_output(self.fx_node)
            codegen.output(name_in_graph_fn, store_pos,
                           name_in_graph_output + '.item()', in_return, idx)

    @classmethod
    def from_value(cls, value: ScalarType, need_guard_check: bool,
                   _helper_functions: HelperFunctions,
                   fx_graph: Optional[FxGraph],
                   extract_code_at_start: list[StorePos]) -> "ScalarVar":
        if id(value) not in dyn.dynamic_vars:
            return cls(value, True, need_guard_check, None,
                       extract_code_at_start)
        else:
            assert fx_graph is not None
            if need_guard_check:
                assert len(extract_code_at_start) > 0
            name = new_name('scalar')
            if not config.get_config('dynshape'):
                fx_node = fx_graph.create_input(torch.tensor(value), name, (),
                                                {}, name)
            else:
                fx_node = fx_graph.create_sym_input(value, name, (), {}, name)
            var = cls.from_value_and_node(value, fx_node, need_guard_check,
                                          extract_code_at_start)
            return var

    @classmethod
    def from_value_and_node(
            cls, value: ScalarType, fx_node: torch.fx.Node,
            need_guard_check: bool,
            extract_code_at_start: list[StorePos]) -> 'ScalarVar':
        var = cls(value, False, need_guard_check, fx_node,
                  extract_code_at_start)
        fx_node.meta["var"] = var
        return var

    def as_fx_node(self) -> NodeArgs:
        if self.value_fix:
            return self.obj
        else:
            return self.fx_node


class NumpyScalarVar(Variable):
    dtype: type
    value: np.generic
    value_fix: bool
    fx_node: Optional[torch.fx.Node]

    def __init__(self, value: np.generic, value_fix: bool,
                 need_guard_check: bool, fx_node: Optional[torch.fx.Node],
                 extract_code_at_start: list[StorePos]) -> None:
        super().__init__(need_guard_check, value, extract_code_at_start)
        if not value_fix:
            assert fx_node is not None
        self.dtype = type(value)
        self.value = value.item()
        self.value_fix = value_fix
        self.fx_node = fx_node

    def make_guard_inner(self, codegen: "GuardFnCodegen",
                         pos: StorePos) -> None:
        codegen.add_check(
            f"isinstance({pos}.item(), {type(self.obj).__name__})")
        if self.value_fix:
            item = self.obj.item()
            if type(item) == float:
                codegen.add_check(f"{pos}.item() == {get_float_string(item)}")
                codegen.add_import("struct")
            elif isinstance(item, str):
                codegen.add_check(f"{pos}.item() == '{item}'")
            else:
                codegen.add_check(f"{pos}.item() == {item}")

    def make_output_inner(self, name_in_graph_fn: str, store_pos: StorePos,
                          codegen: "GraphFnCodegen", in_return: bool,
                          idx: int) -> None:
        if self.value_fix:
            if type(self.obj) == float:
                codegen.output(
                    name_in_graph_fn, store_pos,
                    f"{self.dtype}({get_float_string(self.obj)}) # {self.obj}",
                    in_return, idx)
                codegen.add_import("struct")
            elif isinstance(self.obj, str):
                codegen.output(name_in_graph_fn, store_pos, f"'{self.obj}'",
                               in_return, idx)
            else:
                codegen.output(name_in_graph_fn, store_pos, str(self.obj),
                               in_return, idx)
        else:
            name_in_graph_output = codegen.add_graph_output(self.fx_node)
            codegen.output(name_in_graph_fn, store_pos,
                           name_in_graph_output + '.item()', in_return, idx)

    @classmethod
    def from_value(cls, value: np.generic, need_guard_check: bool,
                   _helper_functions: HelperFunctions,
                   fx_graph: Optional[FxGraph],
                   extract_code_at_start: list[StorePos]) -> "NumpyScalarVar":
        if id(value) not in dyn.dynamic_vars:
            return cls(value, True, need_guard_check, None,
                       extract_code_at_start)
        else:
            assert fx_graph is not None
            assert len(extract_code_at_start) > 0
            name = new_name('np_scalar')
            fx_node = fx_graph.create_input(torch.tensor(value), name, (), {},
                                            name)
            var = cls.from_value_and_node(value, fx_node, need_guard_check,
                                          extract_code_at_start)
            return var

    @classmethod
    def from_value_and_node(
            cls, value: np.generic, fx_node: torch.fx.Node,
            need_guard_check: bool,
            extract_code_at_start: list[StorePos]) -> 'NumpyScalarVar':
        var = cls(value, False, need_guard_check, fx_node,
                  extract_code_at_start)
        fx_node.meta["var"] = var
        return var

    def as_fx_node(self) -> NodeArgs:
        if self.value_fix:
            return self.obj
        else:
            return self.fx_node
