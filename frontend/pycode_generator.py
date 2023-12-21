from typing import Tuple, Any
from itertools import chain
import torch
import torch.fx
from .pycode_writer import PyCodeWriter, new_name, is_valid_name
from .store_pos import StorePos
from .config import get_config


def gen_imports(writer: PyCodeWriter, imports: set[str]) -> None:
    for module_import in imports:
        writer.wl(module_import)


class FnCodegen:
    writer: PyCodeWriter
    imports: set[str]
    key: int
    objs: dict[str, Any]  # name -> obj

    def __init__(self, key: int) -> None:
        self.key = key
        self.writer = PyCodeWriter()
        self.imports = set()
        self.objs = {}

    def add_obj(self, obj: Any, name: str = "", force: bool = False) -> str:
        if force:
            assert name != ""
            assert is_valid_name(name)
            if name in self.objs:
                assert self.objs[name] == obj
            else:
                self.objs[name] = obj
            return name
        else:
            if name == "" or not is_valid_name(name):
                name = new_name("obj")
            elif name in self.objs:
                name = new_name(name)

            self.objs[name] = obj
            return name

    def add_import(self, module_name: str) -> None:
        self.imports.add(f"import {module_name}")

    def add_import_from(self, module_name: str, name: str) -> None:
        self.imports.add(f"from {module_name} import {name}")

    def add_stmt(self, stmt: str) -> None:
        self.writer.wl(stmt)


class GraphFnCodegen(FnCodegen):
    returns: list[Tuple[str, StorePos]]
    graph_inputs: list[tuple[StorePos, bool]]  # (extract_code, to_tensor)
    graph_outputs: list[torch.fx.Node]
    id2name: dict[int, str]  # idx -> name_in_graph_fn

    def __init__(self, key: int) -> None:
        super().__init__(key)
        self.postprossess = PyCodeWriter()
        self.returns = []
        self.graph_inputs = []
        self.graph_outputs = []
        self.id2name = {}

    def output(self, name_in_graph_fn: str, store_pos: StorePos, code: str,
               in_return: bool, idx: int) -> None:
        self.writer.wl(f"{name_in_graph_fn} = {code}")
        if idx != 0:
            self.id2name[idx] = name_in_graph_fn
        if in_return:
            self.returns.append((name_in_graph_fn, store_pos))

    def get_code(self) -> str:
        writer = PyCodeWriter()
        writer.wl(
            f"def ___make_graph_fn({', '.join(chain(('compiled_graph',), self.objs.keys()) )}):"
        )
        writer.block_start()
        gen_imports(writer, self.imports)
        writer.wl(f"def fn(locals):")
        writer.block_start()
        if get_config('debug'):
            writer.wl(
                f"print('running graph_fn (key = {self.key})', locals.keys())")
        # TODO: simplify
        graph_inputs = []
        for x, to_tensor in self.graph_inputs:
            if to_tensor:
                graph_inputs.append(f"torch.tensor({x})")
            else:
                graph_inputs.append(f"{x}.contiguous()")
        writer.wl(f"graph_out = compiled_graph({', '.join(graph_inputs)})"
                 )  # writer.wl(f"print('graph_out', graph_out)")
        writer.write(self.writer.get_code())
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

    def add_graph_input(self,
                        extract_code: StorePos,
                        to_tensor: bool = False) -> None:
        self.graph_inputs.append((extract_code, to_tensor))
        if to_tensor:
            self.add_import("torch")


class GuardFnCodegen(FnCodegen):
    checks: set[tuple[str, StorePos]]
    imports: set[str]
    object_refs: list[Any]  # the reference to objects for id check

    def __init__(self, key: int) -> None:
        super().__init__(key)
        self.checks = set()
        self.imports = set()
        self.object_refs = []

    def add_check(self, check: tuple[str, StorePos]) -> None:
        self.checks.add(check)

    def add_id_check(self, check: tuple[str, StorePos], obj: Any) -> None:
        self.add_check(check)
        self.object_refs.append(obj)

    def get_code(self) -> str:
        writer = PyCodeWriter()
        writer.wl(f"def ___make_guard_fn({', '.join(self.objs.keys())}):")
        writer.block_start()
        gen_imports(writer, self.imports)
        writer.wl(f"def fn(locals):")
        writer.block_start()
        writer.write(f"try:")
        writer.block_start()
        if get_config('debug'):
            writer.wl(
                f"print('running guard_fn (key = {self.key})', locals.keys())")
        writer.write(self.writer.get_code())
        if len(self.checks) == 0:
            writer.wl(f"ok = True")
        else:
            writer.wl(f"ok = True")
            writer.wl(f"missed_check = []")
            for x in self.checks:
                writer.wl(f"if not ({x[0]}):")
                writer.block_start()
                writer.wl(f'''missed_check.append((r"{x[1]}", r"{x[0]}"))''')
                writer.wl(f"ok = False")
                writer.block_end()
        if get_config('debug'):
            writer.wl(f"print('ok = ', ok)")
        writer.block_end()
        writer.wl(f"except Exception as e:")
        writer.block_start()
        writer.wl(f"print('exception in guard_fn:', e, type(e))")
        writer.wl(f'import traceback')
        writer.wl(f"print(traceback.format_exc())")
        writer.wl(f"return (missed_check, False)")
        writer.block_end()
        writer.wl(f"return (missed_check, ok)")
        writer.block_end()
        writer.wl(f"return fn")
        writer.block_end()
        return writer.get_code()

    def get_object_refs(self) -> list[Any]:
        return self.object_refs
