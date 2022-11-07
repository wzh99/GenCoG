from functools import reduce
from typing import Iterable, cast, Dict, Set, NamedTuple

import numpy as np

from .. import Op, TypeSpec, Const, Expr
from ..expr.array import Tuple, List
from ..expr.basic import ExprKind
from ..graph import Graph, GraphVisitor, Output, Operation
from ..graph.base import VertexKind
from ..spec import max_dim, expr_choices


class Diversity:
    def evaluate(self, graph: Graph):
        pass

    @property
    def result(self) -> float:
        raise NotImplemented


class EdgeDiversity(Diversity):
    def __init__(self, ops: Iterable[Op]):
        self._ops = list(ops)
        self._idx_map = {op: i for i, op in enumerate(self._ops)}
        self._mat = np.zeros((len(self._ops),) * 2, dtype=bool)

    def evaluate(self, graph: Graph):
        vis = EdgeMarker(self)
        for o in graph.outputs_:
            vis.visit(o)

    def mark(self, tail: Op, head: Op):
        if tail not in self._ops or head not in self._ops:
            return
        ti, hi = self._idx_map[tail], self._idx_map[head]
        self._mat[ti, hi] = True

    @property
    def result(self) -> float:
        return np.count_nonzero(self._mat) / self._mat.size


class EdgeMarker(GraphVisitor[None]):
    def __init__(self, div: EdgeDiversity):
        super().__init__()
        self._div = div

    def visit_output(self, o: Output):
        self.visit(o.value_.def_)

    def visit_operation(self, opr: Operation):
        for v in opr.inputs_:
            self.visit(v.def_)
            if v.def_.kind == VertexKind.OPR:
                self._div.mark(cast(Operation, v.def_).op_, opr.op_)


class VertexDiversity(Diversity):
    def __init__(self, ops: Iterable[Op]):
        self._ops = list(ops)
        self._specs: Dict[Op, TypeSpec] = {op: op.spec for op in self._ops}
        self._hash: Dict[Op, Set[int]] = {op: set() for op in self._ops}
        self._space: Dict[Op, VertexSpace] = {op: _est_space(op, self._specs[op]) for op in
                                              self._ops}

    def evaluate(self, graph: Graph):
        vis = OperationCollector(self)
        for o in graph.outputs_:
            vis.visit(o)

    def record(self, opr: Operation):
        # Hash input shape
        op = opr.op_
        if op not in self._specs:
            return
        shapes = tuple(tuple(v.type_.shape_) for i, v in enumerate(opr.inputs_)
                       if i not in op.params_)
        h = hash(shapes)

        # Hash attributes
        space = self._space[op]
        for n, v in opr.attrs_:
            if n not in space.attr:
                continue
            if space.attr[n] == 1:
                continue  # skip attributes whose space cannot be estimated
            h ^= hash((n, v))

        # Record hash for this operation
        self._hash[op].add(h)

    def __getitem__(self, op: Op):
        return min(len(self._hash[op]) / self._space[op].total, 1)

    @property
    def op_div(self):
        return np.array([self[op] for op in self._ops])

    @property
    def result(self) -> float:
        return float(np.mean(self.op_div))


class OperationCollector(GraphVisitor[None]):
    def __init__(self, div: VertexDiversity):
        super().__init__()
        self._div = div

    def visit_output(self, o: Output):
        self.visit(o.value_.def_)

    def visit_operation(self, opr: Operation):
        for v in opr.inputs_:
            self.visit(v.def_)
        self._div.record(opr)


class VertexSpace(NamedTuple):
    total: int
    attr: Dict[str, int]


def _est_space(op: Op, spec: TypeSpec) -> VertexSpace:
    # Compute type space estimate for first input
    max_rank = max(spec.first_rank_choices)
    in_space = (max_dim ** max_rank)

    # Double space if the operator accepts more than one non-parameter input
    multi_in = False
    if spec.is_variadic:
        multi_in = True
    else:
        num: int = cast(Const, spec.in_num).val_
        if num - len(op.params_) > 1:
            multi_in = True
    if multi_in:
        in_space *= 2

    # Collect attribute space
    attr_space = 1
    attr_detail = dict()
    for a in spec.attrs:
        num = _est_expr_choices(a.expr_)
        attr_space *= num
        attr_detail[a.name_] = num
    log_attr = int(np.log2(attr_space)) + 1
    est = in_space * log_attr

    return VertexSpace(est, attr=attr_detail)


def _est_expr_choices(expr: Expr):
    # Estimate scalar values
    if expr.type_.is_scalar:
        choices = expr_choices(expr, [])
        return 1 if len(choices) == 0 else len(choices)

    # Estimate array
    if expr.kind == ExprKind.TUPLE:
        tup = cast(Tuple, expr)
        return reduce(lambda acc, f: _est_expr_choices(f) * acc, tup.fields_, 1)
    elif expr.kind == ExprKind.LIST:
        lst = cast(List, expr)
        len_choices = expr_choices(lst.len_, [])
        if len(len_choices) != 0:
            num = max(len_choices)
            return _est_expr_choices(lst.body_) ** num

    return 1
