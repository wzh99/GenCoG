from enum import IntEnum

import numpy as np

from gencog.expr.array import Tuple, List, GetItem, Len, Concat, Slice, Map, ReduceArray, \
    ReduceRange, Filter, InSet, Subset, Perm
from gencog.expr.basic import Const, Var, Arith, Cmp, Not, And, Or, ForAll, Cond, GetAttr, Expr
from gencog.expr.tensor import Num, Shape, Rank
from gencog.expr.visitor import ExprVisitor
from gencog.spec import OpRegistry


def main():
    ops = list(OpRegistry.ops())
    stat = np.empty((len(ops), len(ConstraintKind) + 1), dtype=int)
    for i, op in enumerate(ops):
        counter = ConstraintCounter()
        spec = op.spec
        for a in spec.attrs:
            counter.visit(a.expr_, None)
        counter.visit(spec.in_num, None)
        _count_type_fields(spec.in_ranks, counter)
        _count_type_fields(spec.in_shapes, counter)
        for e in spec.extra:
            counter.visit(e, None)
        counter.visit(spec.out_num, None)
        _count_type_fields(spec.out_ranks, counter)
        _count_type_fields(spec.out_shapes, counter)
        stat[i] = np.concatenate([counter.stat_, [np.sum(counter.stat_)]])
    print('Mean:', np.mean(stat, axis=0))
    print('Min:', np.min(stat, axis=0))
    print('Max:', np.max(stat, axis=0))
    print('\nDetails:')
    for i, op in enumerate(ops):
        print(op.name_, stat[i])


def _count_type_fields(expr: Expr, counter: 'ConstraintCounter'):
    if isinstance(expr, Tuple):
        [counter.visit(f, None) for f in expr.fields_]
    elif isinstance(expr, List):
        counter.visit(expr.body_, None)
    else:
        counter.visit(expr, None)


class ConstraintKind(IntEnum):
    NUMERICAL = 0
    LOGICAL = 1
    STRUCTURAL = 2


class ConstraintCounter(ExprVisitor[None, None]):
    def __init__(self):
        super().__init__()
        self.stat_ = np.zeros((len(ConstraintKind),), dtype=int)

    def count(self, kind: ConstraintKind, num: int = 1):
        self.stat_[kind.value] += num

    def visit_const(self, const: Const, _: None) -> None:
        self.count(ConstraintKind.NUMERICAL)

    def visit_var(self, var: Var, _: None) -> None:
        if var.choices_ is not None or var.ran_ is not None:
            self.count(ConstraintKind.NUMERICAL)

    def visit_arith(self, arith: Arith, _: None) -> None:
        self.count(ConstraintKind.NUMERICAL)

    def visit_cmp(self, cmp: Cmp, _: None) -> None:
        self.count(ConstraintKind.NUMERICAL)

    def visit_not(self, n: Not, _: None) -> None:
        self.count(ConstraintKind.LOGICAL)
        self.visit(n.prop_, None)

    def visit_and(self, a: And, _: None) -> None:
        self.count(ConstraintKind.LOGICAL)
        for e in a.clauses_:
            self.visit(e, None)

    def visit_or(self, o: Or, _: None) -> None:
        self.count(ConstraintKind.LOGICAL)
        for e in o.clauses_:
            self.visit(e, None)

    def visit_forall(self, forall: ForAll, _: None) -> None:
        self.visit(forall.body_, None)

    def visit_cond(self, cond: Cond, _: None) -> None:
        self.count(ConstraintKind.LOGICAL)
        self.visit(cond.tr_br_, None)
        self.visit(cond.fls_br_, None)

    def visit_attr(self, attr: GetAttr, _: None) -> None:
        if attr.type_.is_scalar:
            self.count(ConstraintKind.NUMERICAL)
        else:
            self.count(ConstraintKind.STRUCTURAL)

    def visit_num(self, num: Num, _: None) -> None:
        self.count(ConstraintKind.NUMERICAL)

    def visit_shape(self, shape: Shape, _: None) -> None:
        self.count(ConstraintKind.STRUCTURAL)

    def visit_rank(self, rank: Rank, _: None) -> None:
        self.count(ConstraintKind.NUMERICAL)

    def visit_tuple(self, tup: Tuple, _: None) -> None:
        self.count(ConstraintKind.STRUCTURAL)

    def visit_list(self, lst: List, _: None) -> None:
        self.count(ConstraintKind.STRUCTURAL)

    def visit_getitem(self, getitem: GetItem, _: None) -> None:
        if getitem.type_.is_scalar:
            self.count(ConstraintKind.NUMERICAL)
        else:
            self.count(ConstraintKind.STRUCTURAL)

    def visit_len(self, ln: Len, _: None) -> None:
        self.count(ConstraintKind.NUMERICAL)

    def visit_concat(self, concat: Concat, _: None) -> None:
        self.count(ConstraintKind.STRUCTURAL)

    def visit_slice(self, slc: Slice, _: None) -> None:
        self.count(ConstraintKind.STRUCTURAL)

    def visit_map(self, m: Map, _: None) -> None:
        self.count(ConstraintKind.STRUCTURAL)

    def visit_reduce_array(self, red: ReduceArray, _: None) -> None:
        self.count(ConstraintKind.NUMERICAL)

    def visit_reduce_index(self, red: ReduceRange, _: None) -> None:
        self.count(ConstraintKind.NUMERICAL)

    def visit_filter(self, flt: Filter, _: None) -> None:
        self.count(ConstraintKind.STRUCTURAL)

    def visit_inset(self, inset: InSet, _: None) -> None:
        self.count(ConstraintKind.NUMERICAL)

    def visit_subset(self, subset: Subset, _: None) -> None:
        self.count(ConstraintKind.STRUCTURAL)

    def visit_perm(self, perm: Perm, _: None) -> None:
        self.count(ConstraintKind.STRUCTURAL)


if __name__ == '__main__':
    main()
