from typing import Callable, Optional, cast

from .eval import EvalExpr, EvalFailed
from .store import ValueStore, ArrayNode
from ..expr.array import Tuple, List, GetItem, Len, Concat, Slice, Map, ReduceArray, ReduceIndex, \
    Filter, InSet, Subset
from ..expr.basic import Env, Expr, ExprKind, Const, Var, Symbol, Range, Arith, Cmp, Not, And, Or, \
    ForAll, Cond, GetAttr, to_expr
from ..expr.tensor import Num, Shape, Rank, GetDType
from ..expr.ty import ValueType
from ..expr.visitor import ExprVisitor
from ..util import map_opt


class LowerExpr(ExprVisitor[Env[Expr], Expr]):
    """
    Translate high-level constraint expression to lower level that is suitable for constraint
    solving. Constant folding is also performed if possible.
    """

    def __init__(self, store: ValueStore):
        super().__init__()
        self._store = store
        self._eval = EvalExpr(self._store)

    def visit(self, pre: Expr, env: Env[Expr]) -> Expr:
        post = cast(Expr, super().visit(pre, env))
        return pre if post.kind == ExprKind.DUMMY else post

    def visit_const(self, const: Const, env: Env[Expr]) -> Expr:
        return const

    def visit_var(self, var: Var, env: Env[Expr]) -> Expr:
        return var if env.empty else Var(
            ty=var.type_, ran=map_opt(lambda ran: self.visit_range(ran, env), var.ran_))

    def visit_symbol(self, sym: Symbol, env: Env[Expr]) -> Expr:
        return env[sym]

    def visit_range(self, ran: Range, env: Env[Expr]) -> Range:
        return Range(
            begin=map_opt(lambda e: self.visit(e, env), ran.begin_),
            end=map_opt(lambda e: self.visit(e, env), ran.end_),
            ty=ran.type_
        )

    def visit_arith(self, arith: Arith, env: Env[Expr]) -> Expr:
        return self._try_fold(
            arith, env, lambda: Arith(
                arith.op_, self.visit(arith.lhs_, env), self.visit(arith.rhs_, env),
                ty=arith.type_)
        )

    def visit_cmp(self, cmp: Cmp, env: Env[Expr]) -> Expr:
        return self._try_fold(
            cmp, env, lambda: Cmp(cmp.op_, self.visit(cmp.lhs_, env), self.visit(cmp.rhs_, env))
        )

    def visit_not(self, n: Not, env: Env[Expr]) -> Expr:
        return self._try_fold(n, env, lambda: Not(self.visit(n.prop_, env)))

    def visit_and(self, a: And, env: Env[Expr]) -> Expr:
        return self._try_fold(a, env, lambda: And(*(self.visit(c, env) for c in a.clauses_)))

    def visit_or(self, o: Or, env: Env[Expr]) -> Expr:
        return self._try_fold(o, env, lambda: Or(*(self.visit(c, env) for c in o.clauses_)))

    def visit_forall(self, forall: ForAll, env: Env[Expr]) -> Expr:
        ran = self._try_eval(forall.ran_, env)
        if ran is None:
            return forall
        and_expr = And(*(self.visit(forall.body_, env + (forall.idx_, Const(idx)))
                         for idx in range(ran[0], ran[1])))
        return self.visit(and_expr, env)

    def visit_cond(self, cond: Cond, env: Env[Expr]) -> Expr:
        pred = self._try_eval(cond.pred_, env)
        if pred is None:
            return Cond(self.visit(cond.pred_, env), self.visit(cond.tr_br_, env),
                        self.visit(cond.fls_br_, env))
        return self.visit(cond.tr_br_, env) if pred else self.visit(cond.fls_br_, env)

    def visit_attr(self, attr: GetAttr, env: Env[Expr]) -> Expr:
        return self._store.query_attr(attr.name_).expr

    def visit_num(self, num: Num, env: Env[Expr]) -> Expr:
        node = self._store.query_shape(num.t_kind_)
        return cast(ArrayNode, node).len_.expr

    def visit_shape(self, shape: Shape, env: Env[Expr]) -> Expr:
        idx = self._try_eval(shape.tensor_.idx_, env)
        if idx is None:
            return shape
        node = self._store.query_shape(shape.tensor_.kind_, idx)
        return node.expr

    def visit_rank(self, rank: Rank, env: Env[Expr]) -> Expr:
        idx = self._try_eval(rank.tensor_.idx_, env)
        if idx is None:
            return rank
        node = self._store.query_shape(rank.tensor_.kind_, idx)
        return cast(ArrayNode, node).len_.expr

    def visit_dtype(self, dtype: GetDType, env: Env[Expr]) -> Expr:
        idx = self._try_eval(dtype.tensor_.idx_, env)
        if idx is None:
            return dtype
        node = self._store.query_dtype(dtype.tensor_.kind_, idx)
        return node.expr

    def visit_tuple(self, tup: Tuple, env: Env[Expr]) -> Expr:
        return tup

    def visit_list(self, lst: List, env: Env[Expr]) -> Expr:
        num = self._try_eval(lst.len_, env)
        if num is None:
            return lst
        return Tuple(*(self.visit(lst.body_, env + (lst.idx_, Const(idx)))
                       for idx in range(num)),
                     ty=lst.type_)

    def visit_getitem(self, getitem: GetItem, env: Env[Expr]) -> Expr:
        tup = self.visit(getitem.arr_, env)
        if tup.kind != ExprKind.TUPLE:
            return getitem
        idx = self._try_eval(getitem.idx_, env)
        if idx is None:
            return getitem
        return cast(Tuple, tup).fields_[idx]

    def visit_len(self, ln: Len, env: Env[Expr]) -> Expr:
        tup = self.visit(ln.arr_, env)
        if tup.kind != ExprKind.TUPLE:
            return ln
        return Const(len(cast(Tuple, tup).fields_))

    def visit_concat(self, concat: Concat, env: Env[Expr]) -> Expr:
        fields = []
        for arr in concat.arrays_:
            tup = self.visit(arr, env)
            if tup.kind != ExprKind.TUPLE:
                return concat
            fields.extend(cast(Tuple, tup).fields_)
        return Tuple(*fields, ty=concat.type_)

    def visit_slice(self, slc: Slice, env: Env[Expr]) -> Expr:
        tup = self.visit(slc.arr_, env)
        if tup.kind != ExprKind.TUPLE:
            return slc
        ran = self._try_eval(slc.ran_, env)
        if ran is None:
            return slc
        return Tuple(*cast(Tuple, tup).fields_[ran[0]:ran[1]], ty=slc.type_)

    def visit_map(self, m: Map, env: Env[Expr]) -> Expr:
        tup = self.visit(m.arr_, env)
        if tup.kind != ExprKind.TUPLE:
            return m
        tup = cast(Tuple, tup)
        return Tuple(*(self.visit(m.body_, env + (m.sym_, e)) for e in tup.fields_), ty=m.type_)

    def visit_reduce_array(self, red: ReduceArray, env: Env[Expr]) -> Expr:
        return self._try_fold(red, env, lambda: red)

    def visit_reduce_index(self, red: ReduceIndex, env: Env[Expr]) -> Expr:
        return self._try_fold(red, env, lambda: red)

    def visit_filter(self, flt: Filter, env: Env[Expr]) -> Expr:
        return self._try_fold(flt, env, lambda: flt)

    def visit_inset(self, inset: InSet, env: Env[Expr]) -> Expr:
        return self._visit_sub(inset, env)

    def visit_subset(self, subset: Subset, env: Env[Expr]) -> Expr:
        return self._visit_sub(subset, env)

    def _try_eval(self, expr: Expr, env: Env[Expr]) -> Optional[ValueType]:
        try:
            return self._eval.visit(expr, self._cvt_env(env))
        except EvalFailed:
            return None

    def _try_fold(self, expr: Expr, env: Env[Expr], default: Callable[[], Expr]) -> Expr:
        try:
            v = self._eval.visit(expr, self._cvt_env(env))
            if not expr.type_.is_scalar:
                v = tuple(v)
            return to_expr(v)
        except EvalFailed:
            return default()

    @staticmethod
    def _cvt_env(expr_env: Env[Expr]):
        val_env: Env[ValueType] = Env()
        for sym, expr in expr_env:
            if expr.kind != ExprKind.CONST:
                continue
            const = cast(Const, expr)
            val_env = val_env + (sym, const.val_)
        return val_env
