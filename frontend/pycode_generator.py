from typing import Tuple, Any
from .object_table import ObjectTable
from .pycode_writer import PyCodeWriter
from .variables import Variable


def gen_imports(writer: PyCodeWriter, imports: set[str]) -> None:
    for module_name in imports:
        writer.wl(f"import {module_name}")


class GraphFnCodegen:
    outputs: list[Tuple[str, str]]
    imports: set[str]

    def __init__(self) -> None:
        self.outputs = []
        self.imports = set()

    def output(self, target_name: str, code: str) -> None:
        self.outputs.append((target_name, code))

    def add_import(self, module_name: str) -> None:
        self.imports.add(module_name)

    def get_code(self) -> str:
        writer = PyCodeWriter()
        writer.wl(f"def ___make_graph_fn():")
        writer.block_start()
        gen_imports(writer, self.imports)
        writer.wl(f"def fn(locals):")
        writer.block_start()
        writer.wl(f"print('running graph_fn', locals)")
        for target_name, code in self.outputs:
            writer.wl(f"{target_name} = {code}")
        writer.wl(f"print('graph_fn done', locals)")
        graph_retures = ", ".join(
            [f"{target_name}" for target_name, _ in self.outputs])
        writer.wl(f"return {graph_retures}")
        writer.block_end()
        writer.wl(f"return fn")
        writer.block_end()
        return writer.get_code()

    def get_return_values(self) -> list[str]:
        return [target_name for target_name, _ in self.outputs]


class GuardFnCodegen:
    checks: list[str]
    imports: set[str]

    def __init__(self) -> None:
        self.checks = []
        self.imports = set()

    def add_check(self, check: str) -> None:
        self.checks.append(check)

    def add_import(self, module_name: str) -> None:
        self.imports.add(module_name)

    def get_code(self) -> str:
        writer = PyCodeWriter()
        writer.wl(f"def ___make_guard_fn():")
        writer.block_start()
        gen_imports(writer, self.imports)
        writer.wl(f"def fn(locals):")
        writer.block_start()
        writer.wl(f"print('running guard_fn', locals)")
        if len(self.checks) == 0:
            writer.wl(f"ok = True")
        else:
            writer.wl(
                f"ok = {' and '.join(map(lambda x: f'({x})', self.checks))}")
        writer.wl(f"print('ok = ', ok)")
        writer.wl(f"return ok")
        writer.block_end()
        writer.wl(f"return fn")
        writer.block_end()
        return writer.get_code()
