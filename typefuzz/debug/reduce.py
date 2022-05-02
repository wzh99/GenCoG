from sys import stdout
from typing import List, Dict, Optional, Callable, Set

from numpy.random import Generator, PCG64
from polyleven import levenshtein
from tqdm import tqdm
from tvm import relay, parser, IRModule

from typefuzz.debug.run import build_mod, run_gmod, gen_tensor_value_dict
from typefuzz.graph.relay import tuple_in_ops
from typefuzz.util import filter_none, NameGenerator


class Vertex:
    def __init__(self, expr: relay.Expr, pred: List['Vertex'], tup_in: bool):
        self.expr_ = expr
        self.pred_ = pred
        self.succ_ = []
        for p in self.pred_:
            p.succ_.append(self)
        self.tup_in_ = tup_in
        self.tup_out_indices_ = []

    @property
    def is_tuple_out(self):
        return len(self.tup_out_indices_) > 0


class CaseReducer:
    def __init__(self, code: str, err: str, opt_level: int):
        self.fn_ = parser.parse(code)['main']
        self.err_ = err
        self.opt_level_ = opt_level
        self.graph_ = GraphBuilder().visit_function(self.fn_)

    def has_error(self, mod: IRModule) -> bool:
        raise NotImplementedError

    def reduce(self) -> str:
        # Reduce forward and backward
        vertices = set(self.graph_.vertices_)
        vertices = self._reduce_dir(vertices, lambda v: v.pred_, lambda v: v.succ_)
        vertices = self._reduce_dir(vertices, lambda v: v.succ_, lambda v: v.pred_)

        # Create final function and output result
        outputs = _find_outputs(vertices)
        fn = RelayReducer(vertices, self.graph_.expr2vert_, NameGenerator('rx')).reduce(outputs)
        mod = IRModule.from_expr(fn)
        mod = relay.transform.InferType()(mod)
        return mod.astext()

    def _reduce_dir(self, vertices: Set[Vertex],
                    pred_fn: Callable[[Vertex], List[Vertex]],
                    succ_fn: Callable[[Vertex], List[Vertex]]) -> Set[Vertex]:
        # Count predecessors and find zero-predecessor set
        pred_cnt = {v: len([p for p in pred_fn(v) if p in vertices]) for v in vertices}
        zero_pred = []
        _update_zero_pred(pred_cnt, zero_pred)

        # Iteratively reduce vertices
        progress = tqdm(total=len(vertices), file=stdout)
        while True:
            # Remove one vertex in zero predecessor
            for vert in zero_pred:
                # Generate Relay function
                reduced = vertices.difference([vert])
                outputs = _find_outputs(reduced)
                name_gen = NameGenerator('rx')
                fn = RelayReducer(reduced, self.graph_.expr2vert_, name_gen).reduce(outputs)
                mod = IRModule.from_expr(fn)
                mod = relay.transform.AnnotateSpans()(mod)

                # Test if error occurs
                if not self.has_error(mod):
                    continue

                # Update predecessor count and zero-predecessor set
                zero_pred.remove(vert)
                for s in succ_fn(vert):
                    if s in pred_cnt:
                        pred_cnt[s] -= 1
                _update_zero_pred(pred_cnt, zero_pred)
                vertices = reduced
                progress.update()
                break
            else:
                progress.close()
                break  # no error found, stop iteration

        return vertices


def _update_zero_pred(pred_cnt: Dict[Vertex, int], zero_pred: List[Vertex]):
    added = []
    for v, k in pred_cnt.items():
        if k == 0:
            zero_pred.append(v)
            added.append(v)
    for v in added:
        del pred_cnt[v]


def _find_outputs(vertices: Set[Vertex]):
    return [v for v in vertices if all(s not in vertices for s in v.succ_)]


class CompileReducer(CaseReducer):
    def has_error(self, mod: IRModule) -> bool:
        try:
            build_mod(mod, self.opt_level_)
        except Exception as err:
            return levenshtein(str(err), self.err_) < 100
        else:
            return False


class RunReducer(CaseReducer):
    def has_error(self, mod: IRModule) -> bool:
        try:
            gmod = build_mod(mod, self.opt_level_)
            rng = Generator(PCG64(seed=42))
            inputs = gen_tensor_value_dict(mod['main'].params, rng)
            run_gmod(gmod, inputs)
        except Exception as err:
            return levenshtein(str(err), self.err_) < 100
        else:
            return False


class Graph:
    def __init__(self, zero_succ: List[Vertex], vertices: List[Vertex],
                 vert_map: Dict[relay.Expr, Optional[Vertex]]):
        self.zero_succ_ = zero_succ
        self.vertices_ = vertices
        self.expr2vert_ = vert_map


class RelayReducer(relay.ExprMutator):
    def __init__(self, vertices: Set[Vertex], expr2vert: Dict[relay.Expr, Vertex],
                 name_gen: NameGenerator):
        super().__init__()
        self._vertices = vertices
        self._expr2vert = expr2vert
        self._name_gen = name_gen

    def reduce(self, outputs: List[Vertex]):
        out_exprs = []
        for out in outputs:
            if out.is_tuple_out:
                tup_expr = self.visit(out.expr_)
                for idx in out.tup_out_indices_:
                    out_exprs.append(relay.TupleGetItem(tup_expr, idx))
            else:
                out_exprs.append(self.visit(out.expr_))
        out_tup = relay.Tuple(out_exprs)
        return relay.Function(relay.analysis.free_vars(out_tup), out_tup)

    def visit_call(self, call: relay.Call):
        vert = self._expr2vert[call]
        if vert in self._vertices:
            return super().visit_call(call)
        if vert.is_tuple_out:
            return relay.Tuple([relay.Var(self._name_gen.generate(), type_annotation=ty)
                                for ty in call.checked_type.fields])
        else:
            return relay.Var(self._name_gen.generate(), type_annotation=call.checked_type)

    def visit_tuple_getitem(self, getitem: relay.TupleGetItem):
        getitem = super().visit_tuple_getitem(getitem)
        if isinstance(getitem.tuple_value, relay.Tuple):
            return getitem.tuple_value.fields[getitem.index]
        else:
            return getitem


class GraphBuilder(relay.ExprFunctor):
    def __init__(self):
        super().__init__()
        self._vertices = []

    def visit_function(self, fn: relay.Function):
        body = fn.body
        if isinstance(body, (relay.Call, relay.TupleGetItem)):
            zero_succ = [self.visit(body)]
        elif isinstance(body, relay.Tuple):
            zero_succ = [self.visit(f) for f in body.fields]
        else:
            raise TypeError(f'Unknown body type {type(body)}.')
        return Graph(zero_succ, self._vertices, self.memo_map)

    def visit_var(self, var: relay.Var):
        return None

    def visit_call(self, call: relay.Call):
        # Create predecessors
        op_name = call.op.name
        if op_name in tuple_in_ops:
            args = call.args[0].fields
            tuple_in = True
        else:
            args = call.args
            tuple_in = False
        pred = filter_none([self.visit(a) for a in args])

        # Create vertex
        vert = Vertex(call, pred, tup_in=tuple_in)
        self._vertices.append(vert)
        return vert

    def visit_tuple_getitem(self, getitem: relay.TupleGetItem):
        vert: Vertex = self.visit(getitem.tuple_value)
        vert.tup_out_indices_.append(getitem.index)
        return vert

    def visit_tuple(self, tup: relay.Tuple):
        raise NotImplementedError

    def visit_let(self, _):
        raise NotImplementedError

    def visit_if(self, _):
        raise NotImplementedError

    def visit_global_var(self, _):
        raise NotImplementedError

    def visit_op(self, _):
        raise NotImplementedError

    def visit_constant(self, _):
        raise NotImplementedError

    def visit_ref_create(self, _):
        raise NotImplementedError

    def visit_ref_write(self, _):
        raise NotImplementedError

    def visit_ref_read(self, _):
        raise NotImplementedError

    def visit_constructor(self, _):
        raise NotImplementedError

    def visit_match(self, _):
        raise NotImplementedError
