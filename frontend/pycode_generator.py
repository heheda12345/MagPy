from typing import Tuple, Any
from itertools import chain
import torch
import torch.fx
from .pycode_writer import PyCodeWriter, new_name, is_valid_name
from .store_pos import StorePos
from .variables import Variable


def gen_imports(writer: PyCodeWriter, imports: set[str]) -> None:
    for module_name in imports:
        writer.wl(f"import {module_name}")


class GraphFnCodegen:
    postprossess: PyCodeWriter
    returns: list[Tuple[str, StorePos]]
    imports: set[str]
    graph_inputs: list[StorePos]
    graph_outputs: list[torch.fx.Node]
    objs: dict[str, Any]  # name -> var
    key: int
    id2name: dict[int, str]  # idx -> name_in_graph_fn

    def __init__(self, key: int) -> None:
        self.postprossess = PyCodeWriter()
        self.returns = []
        self.imports = set()
        self.graph_inputs = []
        self.graph_outputs = []
        self.objs = {}
        self.key = key
        self.id2name = {}

    def output(self, name_in_graph_fn: str, store_pos: StorePos, code: str,
               in_return: bool, idx: int) -> None:
        self.postprossess.wl(f"{name_in_graph_fn} = {code}")
        if idx != 0:
            self.id2name[idx] = name_in_graph_fn
        if in_return:
            self.returns.append((name_in_graph_fn, store_pos))

    def add_import(self, module_name: str) -> None:
        self.imports.add(module_name)

    def add_stmt(self, stmt: str) -> None:
        self.postprossess.wl(stmt)

    def get_code(self) -> str:
        writer = PyCodeWriter()
        writer.wl(
            f"def ___make_graph_fn({', '.join(chain(('compiled_graph',), self.objs.keys()) )}):"
        )
        writer.block_start()
        gen_imports(writer, self.imports)
        writer.wl(f"def fn(locals):")
        writer.block_start()
        writer.wl(
            f"print('running graph_fn (key = {self.key})', locals.keys())")
        # TODO: simplify
        writer.wl(
            f"graph_out = compiled_graph({', '.join([str(x) for x in self.graph_inputs])})"
        )  # writer.wl(f"print('graph_out', graph_out)")
        writer.write(self.postprossess.get_code())
        # writer.wl(f"print('graph_fn done', locals)")
        graph_retures = ", ".join(
            f"{target_name}" for target_name, _ in self.returns)
        writer.wl(f"return {graph_retures}")
        writer.block_end()
        writer.wl(f"return fn")
        writer.block_end()
        return writer.get_code()

    def get_return_values(self) -> list[StorePos]:
        return [store_pos for _, store_pos in self.returns]

    def add_graph_output(self, fx_node: torch.fx.Node) -> str:
        self.graph_outputs.append(fx_node)
        return f"graph_out[{len(self.graph_outputs)-1}]"

    def get_graph_outputs(self) -> list[torch.fx.Node]:
        return self.graph_outputs

    def add_graph_input(self, extract_code: StorePos) -> None:
        self.graph_inputs.append(extract_code)

    def add_var(self, var: Any, name: str = "") -> str:
        if name == "" or not is_valid_name(name):
            name = new_name("var")
        elif name in self.objs:
            name = new_name(name)

        self.objs[name] = var
        return name


class GuardFnCodegen:
    checks: list[str]
    imports: set[str]
    vars: dict[str, Variable]  # name -> var
    key: int
    object_refs: list[Any]  # the reference to objects for id check

    def __init__(self, key: int) -> None:
        self.checks = []
        self.imports = set()
        self.vars = {}
        self.key = key
        self.object_refs = []

    def add_check(self, check: str) -> None:
        self.checks.append(check)

    def add_id_check(self, check: str, obj: Any) -> None:
        self.add_check(check)
        self.object_refs.append(obj)

    def add_import(self, module_name: str) -> None:
        self.imports.add(module_name)

    def get_code(self) -> str:
        writer = PyCodeWriter()
        writer.wl(f"def ___make_guard_fn({', '.join(self.vars.keys())}):")
        writer.block_start()
        gen_imports(writer, self.imports)
        writer.wl(f"def fn(locals):")
        writer.block_start()
        writer.write(f"try:")
        writer.block_start()
        writer.wl(
            f"print('running guard_fn (key = {self.key})', locals.keys())")
        if len(self.checks) == 0:
            writer.wl(f"ok = True")
        else:
            writer.wl(
                f"ok = {' and '.join(map(lambda x: f'({x})', self.checks))}")
        writer.wl(f"print('ok = ', ok)")
        writer.block_end()
        writer.wl(f"except Exception as e:")
        writer.block_start()
        writer.wl(f"print('exception in guard_fn:', e, type(e))")
        writer.wl(f'import traceback')
        writer.wl(f"print(traceback.format_exc())")
        writer.wl(f"return False")
        writer.block_end()
        writer.wl(f"return ok")
        writer.block_end()
        writer.wl(f"return fn")
        writer.block_end()
        return writer.get_code()

    def add_var(self, var: Variable, name: str = "") -> str:
        if name == "" or not is_valid_name(name):
            name = new_name("var")
        elif name in self.vars:
            name = new_name(name)

        self.vars[name] = var
        return name

    def get_object_refs(self) -> list[Any]:
        return self.object_refs