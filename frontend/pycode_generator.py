from typing import Tuple
import torch
import torch.fx
from .pycode_writer import PyCodeWriter, new_name, is_valid_name
from .cache import StorePos
from .variables import Variable


def gen_imports(writer: PyCodeWriter, imports: set[str]) -> None:
    for module_name in imports:
        writer.wl(f"import {module_name}")


class GraphFnCodegen:
    outputs: list[Tuple[str, StorePos, str]]
    imports: set[str]
    graph_inputs: list[str]
    graph_outputs: list[torch.fx.Proxy]

    def __init__(self) -> None:
        self.outputs = []
        self.imports = set()
        self.graph_inputs = []
        self.graph_outputs = []

    def output(self, name_in_graph_fn: str, store_pos: StorePos,
               code: str) -> None:
        self.outputs.append((name_in_graph_fn, store_pos, code))

    def add_import(self, module_name: str) -> None:
        self.imports.add(module_name)

    def get_code(self) -> str:
        writer = PyCodeWriter()
        writer.wl(f"def ___make_graph_fn(compiled_graph):")
        writer.block_start()
        gen_imports(writer, self.imports)
        writer.wl(f"def fn(locals):")
        writer.block_start()
        writer.wl(f"print('running graph_fn', locals)")
        # TODO: simplify
        writer.wl(f"graph_out = compiled_graph({', '.join(self.graph_inputs)})")
        writer.wl(f"print('graph_out', graph_out)")
        for target_name, _, code in self.outputs:
            writer.wl(f"{target_name} = {code}")
        writer.wl(f"print('graph_fn done', locals)")
        graph_retures = ", ".join(
            [f"{target_name}" for target_name, _, _ in self.outputs])
        writer.wl(f"return {graph_retures}")
        writer.block_end()
        writer.wl(f"return fn")
        writer.block_end()
        return writer.get_code()

    def get_return_values(self) -> list[StorePos]:
        return [store_pos for _, store_pos, _ in self.outputs]

    def add_graph_output(self, proxy: torch.fx.Proxy) -> str:
        self.graph_outputs.append(proxy)
        return f"graph_out[{len(self.graph_outputs)-1}]"

    def get_graph_outputs(self) -> list[torch.fx.Proxy]:
        return self.graph_outputs

    def add_graph_input(self, extract_code: str) -> None:
        self.graph_inputs.append(extract_code)


class GuardFnCodegen:
    checks: list[str]
    imports: set[str]
    vars: dict[str, Variable]  # name -> var

    def __init__(self) -> None:
        self.checks = []
        self.imports = set()
        self.vars = {}

    def add_check(self, check: str) -> None:
        self.checks.append(check)

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
        writer.wl(f"print('running guard_fn', locals)")
        if len(self.checks) == 0:
            writer.wl(f"ok = True")
        else:
            writer.wl(
                f"ok = {' and '.join(map(lambda x: f'({x})', self.checks))}")
        writer.wl(f"print('ok = ', ok)")
        writer.block_end()
        writer.wl(f"except Exception as e:")
        writer.block_start()
        writer.wl(f"print('exception in graph_fn:', e, type(e))")
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
