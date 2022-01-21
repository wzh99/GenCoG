from typing import Generic, TypeVar, Dict, Callable, Any

from .array import Tuple, List, GetItem, Len, Concat, Slice, Map, ReduceArray, ReduceIndex, Filter, \
    InSet, Subset
from .basic import Expr, ExprKind, Const, Var, Symbol, Range, Arith, Cmp, Not, And, Or, ForAll, \
    Cond, GetAttr
from .tensor import Num, Shape, Rank, GetDType

A = TypeVar('A')
R = TypeVar('R')


class ExprVisitor(Generic[A, R]):
    """
    Base class of constraint expression visitors.
    """

    def __init__(self):
        self._methods: Dict[ExprKind, Callable[[Any, A], R]] = {
            ExprKind.CONST: self.visit_const,
            ExprKind.VAR: self.visit_var,
            ExprKind.SYMBOL: self.visit_symbol,
            ExprKind.RANGE: self.visit_range,
            ExprKind.ARITH: self.visit_arith,
            ExprKind.CMP: self.visit_cmp,
            ExprKind.NOT: self.visit_not,
            ExprKind.AND: self.visit_and,
            ExprKind.OR: self.visit_or,
            ExprKind.FORALL: self.visit_forall,
            ExprKind.COND: self.visit_cond,
            ExprKind.ATTR: self.visit_attr,
            ExprKind.NUM: self.visit_num,
            ExprKind.SHAPE: self.visit_shape,
            ExprKind.RANK: self.visit_rank,
            ExprKind.DTYPE: self.visit_dtype,
            ExprKind.TUPLE: self.visit_tuple,
            ExprKind.LIST: self.visit_list,
            ExprKind.GETITEM: self.visit_getitem,
            ExprKind.LEN: self.visit_len,
            ExprKind.CONCAT: self.visit_concat,
            ExprKind.SLICE: self.visit_slice,
            ExprKind.MAP: self.visit_map,
            ExprKind.REDUCE_ARRAY: self.visit_reduce_array,
            ExprKind.REDUCE_INDEX: self.visit_reduce_index,
            ExprKind.FILTER: self.visit_filter,
            ExprKind.INSET: self.visit_inset,
            ExprKind.SUBSET: self.visit_subset,
        }

    def visit(self, e: Expr, arg: A) -> R:
        return self._methods[e.kind](e, arg)

    def visit_const(self, const: Const, arg: A) -> R:
        pass

    def visit_var(self, var: Var, arg: A) -> R:
        return self._visit_sub(var, arg)

    def visit_symbol(self, sym: Symbol, arg: A) -> R:
        pass

    def visit_range(self, ran: Range, arg: A) -> R:
        return self._visit_sub(ran, arg)

    def visit_arith(self, arith: Arith, arg: A) -> R:
        return self._visit_sub(arith, arg)

    def visit_cmp(self, cmp: Cmp, arg: A) -> R:
        return self._visit_sub(cmp, arg)

    def visit_not(self, n: Not, arg: A) -> R:
        return self._visit_sub(n, arg)

    def visit_and(self, a: And, arg: A) -> R:
        return self._visit_sub(a, arg)

    def visit_or(self, o: Or, arg: A) -> R:
        return self._visit_sub(o, arg)

    def visit_forall(self, forall: ForAll, arg: A) -> R:
        return self._visit_sub(forall, arg)

    def visit_cond(self, cond: Cond, arg: A) -> R:
        return self._visit_sub(cond, arg)

    def visit_attr(self, attr: GetAttr, arg: A) -> R:
        return self._visit_sub(attr, arg)

    def visit_num(self, num: Num, arg: A) -> R:
        return self._visit_sub(num, arg)

    def visit_shape(self, shape: Shape, arg: A) -> R:
        return self._visit_sub(shape, arg)

    def visit_rank(self, rank: Rank, arg: A) -> R:
        return self._visit_sub(rank, arg)

    def visit_dtype(self, dtype: GetDType, arg: A) -> R:
        return self._visit_sub(dtype, arg)

    def visit_tuple(self, tup: Tuple, arg: A) -> R:
        return self._visit_sub(tup, arg)

    def visit_list(self, lst: List, arg: A) -> R:
        return self._visit_sub(lst, arg)

    def visit_getitem(self, getitem: GetItem, arg: A) -> R:
        return self._visit_sub(getitem, arg)

    def visit_len(self, ln: Len, arg: A) -> R:
        return self._visit_sub(ln, arg)

    def visit_concat(self, concat: Concat, arg: A) -> R:
        return self._visit_sub(concat, arg)

    def visit_slice(self, slc: Slice, arg: A) -> R:
        return self._visit_sub(slc, arg)

    def visit_map(self, m: Map, arg: A) -> R:
        return self._visit_sub(m, arg)

    def visit_reduce_array(self, red: ReduceArray, arg: A) -> R:
        return self._visit_sub(red, arg)

    def visit_reduce_index(self, red: ReduceIndex, arg: A) -> R:
        return self._visit_sub(red, arg)

    def visit_filter(self, flt: Filter, arg: A) -> R:
        return self._visit_sub(flt, arg)

    def visit_inset(self, inset: InSet, arg: A) -> R:
        return self._visit_sub(inset, arg)

    def visit_subset(self, subset: Subset, arg: A) -> R:
        return self._visit_sub(subset, arg)

    def _visit_sub(self, expr: Expr, arg: A):
        for s in expr.sub_expr_:
            self.visit(s, arg)
        return None
