from typing import Any, Callable, Dict, Optional, Tuple, Union
from functools import partial
import copy
import collections
import torch
import torch.fx
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch._guards import Source
import torch._inductor.compile_fx
import torch._dynamo.backends.torchxla
import torch.fx.immutable_collections as fx_immutable
from torch._dispatch.python import enable_python_dispatcher
from torch import SymInt, SymFloat, SymBool
from torch.fx.experimental.symbolic_shapes import Symbol
from sympy.printing.str import StrPrinter
import sympy
from .no_preload import NO_LD_PRELOAD_CTX
from . import config
from .utils import ScalarType
from .pycode_generator import GuardFnCodegen
from .store_pos import StorePos, StoreNegate, StoreInAttr, StoreInIndex, voidpos
from . import variables as vs

BaseArgumentTypes = Union[
    str,
    int,
    float,
    bool,
    complex,
    #   torch.dtype, torch.Tensor, torch.device, torch.memory_format,
    #   torch.layout, torch._ops.OpOverload
]

NodeArgs = Union[BaseArgumentTypes, torch.fx.Node]


def backend_compile(gm: torch.fx.GraphModule,
                    example_inputs: list[torch.Tensor]) -> Any:
    backend = config.get_config('backend')
    if callable(backend):
        return backend(gm, example_inputs)
    elif backend == 'eager':
        return gm
    elif backend == 'inductor':

        def fetch_attr(gm, target: str) -> Any:
            target_atoms = target.split('.')
            attr_itr = gm
            for i, atom in enumerate(target_atoms):
                if not hasattr(attr_itr, atom):
                    raise RuntimeError(
                        f"Node referenced nonexistent target {'.'.join(target_atoms[:i])}"
                    )
                attr_itr = getattr(attr_itr, atom)
            return attr_itr

        def eager_due_to_inductor_bug(node: torch.fx.Node) -> bool:

            if node.op == 'call_module':
                module = fetch_attr(gm, node.target)
                if isinstance(module, torch.nn.RNNBase):
                    return True
            return False

        for node in gm.graph.nodes:
            if eager_due_to_inductor_bug(node):
                print("fall back to eager due to", node)
                return gm
        return torch._inductor.compile_fx.compile_fx(gm, example_inputs)
    elif backend == 'xla':
        return torch._dynamo.backends.torchxla.aot_torchxla_trace_once(
            gm, example_inputs)
    else:
        raise RuntimeError(f"Unknown backend: {backend}")


def guard_check_shapeenv(inputs: list[torch.Tensor], fake_inputs: list[Any],
                         shape_env: ShapeEnv) -> bool:
    symbol2value: dict[Symbol, Any] = {}
    for fake_input, input in zip(fake_inputs, inputs):
        if isinstance(fake_input, torch._subclasses.FakeTensor):
            assert isinstance(input, torch.Tensor)
            if len(input.shape) != len(fake_input.shape):
                return False
            for symbol, value in zip(fake_input.shape, input.shape):
                expr = symbol.node.expr
                if expr in symbol2value:
                    if symbol2value[expr] != value:
                        print("false due to shape", fake_input.shape,
                              input.shape)
                        print("symbol2value", symbol2value[expr])
                        return False
                else:
                    symbol2value[expr] = value
        else:
            raise NotImplementedError
    for guard in shape_env.guards:
        val = guard.expr.subs(symbol2value)
        if not (val is sympy.true):
            print("guard fail", guard, symbol2value)
            return False
    return True


class ShapeGuardPrinter(StrPrinter):  # type: ignore[misc]

    def __init__(self, symbol_to_source: Dict[Symbol, list[StorePos]]):
        super().__init__()
        self.symbol_to_source = symbol_to_source

    def _print_Symbol(self, expr: Symbol) -> str:
        assert isinstance(expr, Symbol), str(type(expr))
        assert expr in self.symbol_to_source, (
            f"{expr} (could be from {[s.name() for s in expr.sources]}) "
            f"not in {self.symbol_to_source}")
        return str(self.symbol_to_source[expr][0])


class FxGraph:
    root: torch.nn.Module
    result_graph: torch.fx.Graph
    mark_written_fn: Callable[[], None]
    fake_mode: torch._subclasses.FakeTensorMode
    example_inputs: list[tuple[torch.Tensor, str]]

    def __init__(self, root: torch.nn.Module,
                 mark_written_fn: Callable[[], None]) -> None:
        self.root = root
        self.result_graph = torch.fx.Graph(root)
        self.mark_written_fn = mark_written_fn
        self.dynamic_shape = config.get_config('dynshape')
        self.fake_mode = torch._subclasses.FakeTensorMode(
            shape_env=ShapeEnv() if self.dynamic_shape else None,
            # allow_non_fake_inputs=True
        )
        self.example_inputs = []

    def infer_fake_value(self, node: torch.fx.Node) -> None:

        def wrap_fake_exception(fn: Callable[[], Any]) -> Any:
            try:
                return fn()
            except torch._subclasses.UnsupportedFakeTensorException as e:
                msg = f"Unsupported: {e.reason} with fake tensor propagation."
                raise NotImplementedError(msg) from e

        def as_fake_args_kwargs(
                args: Tuple[Any, ...],
                kwargs: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:

            def as_fake(arg: Any) -> Any:
                if isinstance(arg, (tuple, list)):
                    return fx_immutable.immutable_list(
                        [as_fake(x) for x in arg])
                if isinstance(arg, slice):
                    return slice(as_fake(arg.start), as_fake(arg.stop),
                                 as_fake(arg.step))
                if isinstance(arg, torch.fx.Node):
                    return arg.meta["fake"]
                else:
                    return arg

            fake_args = tuple(as_fake(arg) for arg in args)
            fake_kwargs = {k: as_fake(v) for k, v in kwargs.items()}
            return fake_args, fake_kwargs

        def fetch_attr(target: str) -> Any:
            target_atoms = target.split('.')
            attr_itr = self.root
            for i, atom in enumerate(target_atoms):
                if not hasattr(attr_itr, atom):
                    raise RuntimeError(
                        f"Node referenced nonexistent target {'.'.join(target_atoms[:i])}"
                    )
                attr_itr = getattr(attr_itr, atom)
            return attr_itr

        fake_args, fake_kwargs = as_fake_args_kwargs(node.args, node.kwargs)
        fake: Any = None
        op = node.op
        assert op not in ("placeholder", "output")
        if op == "get_attr":
            with self.fake_mode, enable_python_dispatcher():
                param = fetch_attr(node.target)
                fake = self.fake_mode.from_tensor(param, static_shapes=True)
        elif op == "call_function":
            with self.fake_mode, enable_python_dispatcher():
                fake = node.target(*fake_args, **fake_kwargs)
        elif op == "call_method":
            with self.fake_mode, enable_python_dispatcher():
                fake = getattr(fake_args[0], node.target)(*fake_args[1:],
                                                          **fake_kwargs)
        elif op == "call_module":
            module = fetch_attr(node.target)
            with torch._subclasses.fake_tensor.FakeCopyMode(self.fake_mode):
                fake_module = wrap_fake_exception(lambda: copy.deepcopy(module))
            with self.fake_mode, enable_python_dispatcher():
                fake = fake_module(*fake_args, **fake_kwargs)
        else:
            raise RuntimeError(f"Unknown target: {node.target}")
        node.meta["fake"] = fake

    def create_node(
        self,
        kind: str,
        target: torch.fx.node.Target,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
        name: Optional[str] = None,
        type_expr: Optional[Any] = None,
    ) -> torch.fx.Node:
        self.mark_written_fn()
        result_node = self.result_graph.create_node(kind, target, args, kwargs,
                                                    name, type_expr)
        if self.dynamic_shape:
            if kind not in ("placeholder", "output"):
                self.infer_fake_value(result_node)
        return result_node

    def create_input(
        self,
        value: torch.Tensor,
        target: torch.fx.node.Target,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
        name: str,
        type_expr: Optional[Any] = None,
    ) -> torch.fx.Node:
        fake_tensor = self.fake_mode.from_tensor(
            value, static_shapes=not self.dynamic_shape)
        self.mark_written_fn()
        self.example_inputs.append((fake_tensor, name))
        node = self.create_node("placeholder", target, args, kwargs, name,
                                type_expr)
        node.meta["fake"] = fake_tensor
        return node

    def create_sym_input(
        self,
        value: ScalarType,
        target: torch.fx.node.Target,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
        name: str,
        type_expr: Optional[Any] = None,
    ) -> torch.fx.Node:
        symbol = self.fake_mode.shape_env.create_symbol(value, Source())
        fake = self.fake_mode.shape_env.create_symintnode(symbol, hint=value)
        self.mark_written_fn()
        self.example_inputs.append((fake, name))
        node = self.create_node("placeholder", target, args, kwargs, name,
                                type_expr)
        node.meta["fake"] = fake
        return node

    def set_output_nodes(self, output_nodes: list[torch.fx.Node]) -> None:
        for node in self.result_graph.nodes:
            assert node.op != "output"
        self.result_graph.output(tuple(output_nodes))

    def compile(
        self,
    ) -> Any:  # heheda: shoud be Callable[..., Any], but I cannot pass mypy check
        model = torch.fx.GraphModule(self.root, self.result_graph)
        model.recompile()
        with NO_LD_PRELOAD_CTX():
            compiled_fn = backend_compile(model, [
                x[0].contiguous() if isinstance(x[0], torch.Tensor) else x[0]
                for x in self.example_inputs
            ])
        assert callable(compiled_fn)
        if self.fake_mode.shape_env is not None:
            print("shape_env guards", self.fake_mode.shape_env.format_guards())
        # TODO: add backend compiler
        return compiled_fn

    def get_inputs(self) -> list[torch.fx.Node]:
        return [x for x in self.result_graph.nodes if x.op == "placeholder"]

    def make_shape_env_guard(self, codegen: GuardFnCodegen) -> None:
        fake_inputs: list[torch.FakeTensor] = []
        poses: list[StorePos] = []
        for node in self.result_graph.nodes:
            if node.op == "placeholder":
                fake = node.meta["fake"]
                fake_inputs.append(fake)
                var = node.meta["var"]
                assert isinstance(var, (vs.TensorVar, vs.ScalarVar))
                pos = var.extract_code_at_start[0]
                poses.append(pos)
        self.produce_guards(fake_inputs, poses, codegen)

    # modified from torch produce_guards
    def produce_guards(self, placeholders: list[Any], sources: list[StorePos],
                       codegen: GuardFnCodegen) -> None:
        import math
        import operator
        SYMPY_INTERP = {
            'Eq': operator.eq,
            'Ne': operator.ne,
            'Gt': operator.gt,
            'Lt': operator.lt,
            'Le': operator.le,
            'Ge': operator.ge,
            'Min': min,
            'Max': max,
            'Mod': operator.mod,
            'FloorDiv': operator.floordiv,
            'TrueDiv': operator.truediv,
            'floor': math.floor,
            'ceiling': math.ceil,
        }
        for k, v in SYMPY_INTERP.items():
            codegen.add_obj(v, k, force=True)
        input_guards = []
        symbol_to_source = collections.defaultdict(list)

        def track_symint(source: StorePos, val: Any) -> None:
            if isinstance(val, SymInt):
                s = val.node.expr

                if isinstance(s, sympy.Symbol):
                    symbol_to_source[s].append(source)
                elif isinstance(-s, sympy.Symbol):
                    symbol_to_source[-s].append(StoreNegate(source))

                input_guards.append((source, s))
            else:
                input_guards.append((source, sympy.Integer(val)))

        for t, source in zip(placeholders, sources):
            assert isinstance(source, StorePos)
            if t is None:
                continue
            if isinstance(t, SymInt):
                track_symint(source, t)
                continue
            assert isinstance(t, torch.Tensor)
            for i, s in enumerate(t.size()):
                track_symint(
                    StoreInIndex(StoreInAttr(source, 0, 'size()'), 0, i), s)

        for source, expr in input_guards:
            # Small optimization
            if (isinstance(expr, Symbol) and expr in symbol_to_source and
                    source == symbol_to_source[expr][0]):
                continue
            sexpr = ShapeGuardPrinter(symbol_to_source).doprint(expr)
            codegen.add_check((f"{source} == {sexpr}", voidpos()))
        for g, tb in self.fake_mode.shape_env.guards:
            print("guard", g)
            if self.fake_mode.shape_env._maybe_evaluate_static(g) is not None:
                print("maybe static")
                continue
            print("before simplify", g)
            g = self.fake_mode.shape_env.simplify(g)
            print("after simplify", g)
            try:
                codegen.add_check(
                    (ShapeGuardPrinter(symbol_to_source).doprint(g), voidpos()))
            except Exception:
                print(f"Failing guard allocated at: \n{tb}")
                raise

        for sources in symbol_to_source.values():
            assert sources
            codegen.add_check((f"{sources[0]} != 0", voidpos()))
            codegen.add_check((f"{sources[0]} != 1", voidpos()))


frame_root: dict[int, torch.nn.Module] = {}


def set_frame_root(frame_id: int, root: Any) -> None:
    if isinstance(root, torch.nn.Module):
        root_module = root
    elif hasattr(root, '__self__') and isinstance(root.__self__,
                                                  torch.nn.Module):
        root_module = root.__self__
    else:
        root_module = torch.nn.Module()

    frame_root[frame_id] = root_module


def get_frame_root(frame_id: int) -> Any:
    return frame_root[frame_id]


def is_leaf_module(m: torch.nn.Module) -> bool:
    return ((m.__module__.startswith("torch.nn") or
             m.__module__.startswith("torch.autograd.nn")) and
            not isinstance(m, torch.nn.Sequential))


def reset() -> None:
    global frame_root
    frame_root = {}
