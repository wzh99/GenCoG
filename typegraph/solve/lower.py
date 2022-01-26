from .store import ValueStore
from ..expr import ExprVisitor
from ..expr.array import Tuple, List, GetItem, Len, Concat, Slice, Map, ReduceArray, ReduceIndex, \
    Filter, InSet, Subset
from ..expr.basic import Env, Expr, Const, Var, Symbol, Range, Arith, Cmp, Not, And, Or, ForAll, \
    Cond, GetAttr
from ..expr.tensor import Num, Shape, Rank, GetDType
from ..expr.ty import ValueType
from typing import cast
from ..util import map_opt


class LowerFailed(Exception):
    """
    Current expression can not be lowered. This exception is for early exit when lowering is not
    feasible. It is usually not serious.
    """


class LowerExpr(ExprVisitor[Env[ValueType], Expr]):
    """
    Translate high-level constraint expressions to lower level. Each call of `visit` only lower
    the outmost layer(s) of expression.
    """

    def __init__(self, store: ValueStore):
        super().__init__()
        self._store = store

    def visit(self, e: Expr, env: Env[ValueType]) -> Expr:
        return super().visit(e, env)

    def visit_const(self, const: Const, env: Env[ValueType]) -> Expr:
        return const

    def visit_var(self, var: Var, env: Env[ValueType]) -> Expr:
        return Var(ty=var.type_, ran=cast(Range, self.visit(var.ran_, env)))

    def visit_symbol(self, sym: Symbol, env: Env[ValueType]) -> Expr:
        return Const(env[sym])

    def visit_range(self, ran: Range, env: Env[ValueType]) -> Expr:
        return Range(
            begin=map_opt(lambda e: self.visit(e, env), ran.begin_),
            end=map_opt(lambda e: self.visit(e, env), ran.end_),
            ty=ran.type_
        )

    def visit_arith(self, arith: Arith, env: Env[ValueType]) -> Expr:
        return Arith(arith.op_, self.visit(arith.lhs_, env), self.visit(arith.rhs_, env),
                     ty=arith.type_)

    def visit_cmp(self, cmp: Cmp, env: Env[ValueType]) -> Expr:
        return self._visit_sub(cmp, env)

    def visit_not(self, n: Not, env: Env[ValueType]) -> Expr:
        return self._visit_sub(n, env)

    def visit_and(self, a: And, env: Env[ValueType]) -> Expr:
        return self._visit_sub(a, env)

    def visit_or(self, o: Or, env: Env[ValueType]) -> Expr:
        return self._visit_sub(o, env)

    def visit_forall(self, forall: ForAll, env: Env[ValueType]) -> Expr:
        return self._visit_sub(forall, env)

    def visit_cond(self, cond: Cond, env: Env[ValueType]) -> Expr:
        return self._visit_sub(cond, env)

    def visit_attr(self, attr: GetAttr, env: Env[ValueType]) -> Expr:
        return self._visit_sub(attr, env)

    def visit_num(self, num: Num, env: Env[ValueType]) -> Expr:
        return self._visit_sub(num, env)

    def visit_shape(self, shape: Shape, env: Env[ValueType]) -> Expr:
        return self._visit_sub(shape, env)

    def visit_rank(self, rank: Rank, env: Env[ValueType]) -> Expr:
        return self._visit_sub(rank, env)

    def visit_dtype(self, dtype: GetDType, env: Env[ValueType]) -> Expr:
        return self._visit_sub(dtype, env)

    def visit_tuple(self, tup: Tuple, env: Env[ValueType]) -> Expr:
        return self._visit_sub(tup, env)

    def visit_list(self, lst: List, env: Env[ValueType]) -> Expr:
        return self._visit_sub(lst, env)

    def visit_getitem(self, getitem: GetItem, env: Env[ValueType]) -> Expr:
        return self._visit_sub(getitem, env)

    def visit_len(self, ln: Len, env: Env[ValueType]) -> Expr:
        return self._visit_sub(ln, env)

    def visit_concat(self, concat: Concat, env: Env[ValueType]) -> Expr:
        return self._visit_sub(concat, env)

    def visit_slice(self, slc: Slice, env: Env[ValueType]) -> Expr:
        return self._visit_sub(slc, env)

    def visit_map(self, m: Map, env: Env[ValueType]) -> Expr:
        return self._visit_sub(m, env)

    def visit_reduce_array(self, red: ReduceArray, env: Env[ValueType]) -> Expr:
        return self._visit_sub(red, env)

    def visit_reduce_index(self, red: ReduceIndex, env: Env[ValueType]) -> Expr:
        return self._visit_sub(red, env)

    def visit_filter(self, flt: Filter, env: Env[ValueType]) -> Expr:
        return self._visit_sub(flt, env)

    def visit_inset(self, inset: InSet, env: Env[ValueType]) -> Expr:
        return self._visit_sub(inset, env)

    def visit_subset(self, subset: Subset, env: Env[ValueType]) -> Expr:
        return self._visit_sub(subset, env)
