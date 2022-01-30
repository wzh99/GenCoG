import typing as t
from functools import reduce
from itertools import chain, islice
from typing import Union, Iterator, Optional, Callable, cast

from .store import ValueStore, ArrayNode
from ..expr.array import Tuple, List, GetItem, Len, Concat, Slice, Map, ReduceArray, ReduceIndex, \
    Filter, InSet, Subset
from ..expr.basic import Env, Expr, ExprKind, Const, Var, Symbol, Range, Arith, Cmp, Not, And, Or, \
    ForAll, Cond, GetAttr, Dummy, to_expr
from ..expr.tensor import Num, Shape, Rank, GetDType, TensorKind
from ..expr.ty import ValueType
from ..expr.visitor import ExprVisitor
from ..util import map_opt


class EvalError(Exception):
    """
    Current expression cannot be evaluated. This exception is for early exit when evaluation is not
    feasible. It is usually not serious.
    """

    def __init__(self, expr: Expr, msg: str):
        self.expr_ = expr
        self.msg_ = msg


RangeValueType = Union[int, float, None]
RangeType = Union[t.Tuple[RangeValueType, RangeValueType]]
ResultType = Union[ValueType, Iterator]


class EvalExpr(ExprVisitor[Env[ValueType], ResultType]):
    """
    Evaluate a constraint expression.
    """

    def __init__(self, store: ValueStore):
        super().__init__()
        self._store = store

    def evaluate(self, e: Expr):
        return self.visit(e, Env())

    def visit(self, e: Expr, env: Env[ValueType]) -> ResultType:
        v = super().visit(e, env)
        if v is None:
            # Only query from value store will produce None.
            raise EvalError(e, 'Cannot query from value store.')
        return v

    def visit_const(self, const: Const, env: Env[ValueType]) -> ResultType:
        return const.val_

    def visit_var(self, var: Var, env: Env[ValueType]) -> ResultType:
        v = self._store.query_var(var)
        if v is None:
            raise EvalError(var, 'Variable not solved before.')
        return v

    def visit_symbol(self, sym: Symbol, env: Env[ValueType]) -> ResultType:
        if sym not in env:
            raise EvalError(sym, 'Symbol is not found in environment.')
        return env[sym]

    def visit_range(self, ran: Range, env: Env[ValueType]) -> RangeType:
        return (map_opt(lambda e: self.visit(e, env), ran.begin_),
                map_opt(lambda e: self.visit(e, env), ran.end_))

    def visit_arith(self, arith: Arith, env: Env[ValueType]) -> ResultType:
        lv = self.visit(arith.lhs_, env)
        rv = self.visit(arith.rhs_, env)
        return Arith.op_funcs[arith.op_][arith.type_](lv, rv)

    def visit_cmp(self, cmp: Cmp, env: Env[ValueType]) -> ResultType:
        lv = self.visit(cmp.lhs_, env)
        rv = self.visit(cmp.rhs_, env)
        return Cmp.op_funcs[cmp.op_][cmp.lhs_.type_](lv, rv)

    def visit_not(self, n: Not, env: Env[ValueType]) -> ResultType:
        return not self.visit(n.prop_, env)

    def visit_and(self, a: And, env: Env[ValueType]) -> ResultType:
        return all(map(lambda c: self.visit(c, env), a.clauses_))

    def visit_or(self, o: Or, env: Env[ValueType]) -> ResultType:
        return any(map(lambda c: self.visit(c, env), o.clauses_))

    def visit_forall(self, forall: ForAll, env: Env[ValueType]) -> ResultType:
        begin, end = self.visit_range(forall.ran_, env)
        return (self.visit(forall.body_, env + (forall.idx_, idx)) for idx in range(begin, end))

    def visit_cond(self, cond: Cond, env: Env[ValueType]) -> ResultType:
        pred = self.visit(cond.pred_, env)
        return self.visit(cond.tr_br_, env) if pred else self.visit(cond.fls_br_, env)

    def visit_attr(self, attr: GetAttr, env: Env[ValueType]) -> ResultType:
        return self._store.query_attr(attr.name_).value

    def visit_dummy(self, dum: Dummy, env: Env[ValueType]) -> ResultType:
        raise EvalError(dum, 'Dummy expression cannot be evaluated.')

    def visit_num(self, num: Num, env: Env[ValueType]) -> ResultType:
        node = self._store.query_shape(num.t_kind_)
        return cast(ArrayNode, node).len_.value

    def visit_shape(self, shape: Shape, env: Env[ValueType]) -> ResultType:
        idx = self.visit(shape.tensor_.idx_, env)
        node = self._get_tensor_shape_node(shape.tensor_.kind_, idx, shape)
        return node.value

    def visit_rank(self, rank: Rank, env: Env[ValueType]) -> ResultType:
        idx = self.visit(rank.tensor_.idx_, env)
        node = self._get_tensor_shape_node(rank.tensor_.kind_, idx, rank)
        return node.len_.value

    def _get_tensor_shape_node(self, kind: TensorKind, idx: int, e: Expr):
        node = self._store.query_shape(kind, idx)
        if node is None:
            raise EvalError(
                e, f'Shape of {kind.name} tensor {idx} is undefined.'
            )
        return cast(ArrayNode, node)

    def visit_dtype(self, dtype: GetDType, env: Env[ValueType]) -> ResultType:
        kind = dtype.tensor_.kind_
        idx = self.visit(dtype.tensor_.idx_, env)
        node = self._store.query_dtype(kind, idx)
        if node is None:
            raise EvalError(
                dtype, f'Data type of {kind.name} tensor {idx} is undefined.'
            )
        return node.value

    def visit_tuple(self, tup: Tuple, env: Env[ValueType]) -> ResultType:
        return (self.visit(f, env) for f in tup.fields_)

    def visit_list(self, lst: List, env: Env[ValueType]) -> ResultType:
        return (self.visit(lst.body_, env + (lst.idx_, idx))
                for idx in range(self.visit(lst.len_, env)))

    def visit_getitem(self, getitem: GetItem, env: Env[ValueType]) -> ResultType:
        arr = tuple(self.visit(getitem.arr_, env))
        idx = self.visit(getitem.idx_, env)
        if idx >= len(arr):
            raise EvalError(getitem, 'Index out of bound.')
        return arr[idx]

    def visit_len(self, ln: Len, env: Env[ValueType]) -> ResultType:
        return len(tuple(self.visit(ln.arr_, env)))

    def visit_concat(self, concat: Concat, env: Env[ValueType]) -> ResultType:
        return chain(*(self.visit(arr, env) for arr in concat.arrays_))

    def visit_slice(self, slc: Slice, env: Env[ValueType]) -> ResultType:
        begin, end = self.visit_range(slc.ran_, env)
        return islice(self.visit(slc.arr_, env), begin, end)

    def visit_map(self, m: Map, env: Env[ValueType]) -> ResultType:
        return map(lambda v: self.visit(m.body_, env + (m.sym_, v)), self.visit(m.arr_, env))

    def visit_reduce_array(self, red: ReduceArray, env: Env[ValueType]) -> ResultType:
        func = Arith.op_funcs[red.op_][red.type_]
        return reduce(lambda acc, v: func(acc, v), self.visit(red.arr_, env),
                      self.visit(red.init_, env))

    def visit_reduce_index(self, red: ReduceIndex, env: Env[ValueType]) -> ResultType:
        begin, end = self.visit_range(red.ran_, env)
        func = Arith.op_funcs[red.op_][red.type_]
        return reduce(lambda acc, idx: func(acc, self.visit(red.body_, env + (red.idx_, idx))),
                      range(begin, end), self.visit(red.init_, env))

    def visit_filter(self, flt: Filter, env: Env[ValueType]) -> ResultType:
        return filter(lambda v: self.visit(flt.pred_, env + (flt.sym_, v)),
                      self.visit(flt.arr_, env))

    def visit_inset(self, inset: InSet, env: Env[ValueType]) -> ResultType:
        return self.visit(inset.elem_, env) in tuple(self.visit(inset.set_, env))

    def visit_subset(self, subset: Subset, env: Env[ValueType]) -> ResultType:
        sup = self.visit(subset.sup_, env)
        return all(map(lambda v: v in sup, self.visit(subset.sub_, env)))


class PartialEval(ExprVisitor[Env[Expr], Expr]):
    """
    Perform partial evaluation on a constraint expression to make it suitable for constraint
    solving.
    """

    def __init__(self, store: ValueStore):
        super().__init__()
        self._store = store
        self._eval = EvalExpr(self._store)

    def transform(self, e: Expr) -> Expr:
        return self.visit(e, Env())

    def visit(self, pre: Expr, env: Env[Expr]) -> Expr:
        post = cast(Expr, super().visit(pre, env))
        return pre if post.kind == ExprKind.DUMMY else post

    def visit_const(self, const: Const, env: Env[Expr]) -> Expr:
        return const

    def visit_var(self, var: Var, env: Env[Expr]) -> Expr:
        if var.tmpl_:  # create new variable for template
            return Var(ty=var.type_,
                       ran=map_opt(lambda ran: self.visit_range(ran, env), var.ran_))
        else:
            if var.ran_ is not None:  # non-template variable must keep its original object id
                var.ran_ = self.visit_range(var.ran_, env)
            return var

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
        clauses = []
        for c in a.clauses_:
            post = self._try_fold(c, env, lambda: self.visit(c, env))
            if post.kind == Const:
                const = cast(Const, post)
                if const.val_ is False:
                    return Const(False)
            clauses.append(post)
        return And(*clauses)

    def visit_or(self, o: Or, env: Env[Expr]) -> Expr:
        clauses = []
        for c in o.clauses_:
            post = self._try_fold(c, env, lambda: self.visit(c, env))
            if post.kind == Const:
                const = cast(Const, post)
                if const.val_ is True:
                    return Const(True)
            clauses.append(post)
        return Or(*clauses)

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
            return cond
        return self.visit(cond.tr_br_, env) if pred else self.visit(cond.fls_br_, env)

    def visit_attr(self, attr: GetAttr, env: Env[Expr]) -> Expr:
        return self._store.query_attr(attr.name_).expr

    def visit_dummy(self, dum: Dummy, env: Env[Expr]) -> Expr:
        return dum

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
        return self._try_fold(
            red, env, lambda: ReduceArray(
                self.visit(red.arr_, env), red.op_, self.visit(red.init_, env), ty=red.type_
            )
        )

    def visit_reduce_index(self, red: ReduceIndex, env: Env[Expr]) -> Expr:
        return self._try_fold(
            red, env, lambda: ReduceIndex(
                self.visit_range(red.ran_, env), red.op_, self.visit(red.init_, env),
                idx=red.idx_, body=self.visit(red.body_, env + (red.idx_, red.idx_)),
                ty=red.type_
            )
        )

    def visit_filter(self, flt: Filter, env: Env[Expr]) -> Expr:
        return self._try_fold(
            flt, env, lambda: Filter(
                self.visit(flt.arr_, env), sym=flt.sym_,
                pred=self.visit(flt.pred_, env + (flt.sym_, flt.sym_))
            )
        )

    def visit_inset(self, inset: InSet, env: Env[Expr]) -> Expr:
        return self._try_fold(
            inset, env, lambda: InSet(
                self.visit(inset.elem_, env), self.visit(inset.set_, env)
            )
        )

    def visit_subset(self, subset: Subset, env: Env[Expr]) -> Expr:
        return self._try_fold(
            subset, env, lambda: Subset(
                self.visit(subset.sub_, env), self.visit(subset.sup_, env)
            )
        )

    def _try_eval(self, expr: Expr, env: Env[Expr]) -> Optional[ValueType]:
        try:
            return self._eval.visit(expr, self._cvt_env(env))
        except EvalError:
            return None

    def _try_fold(self, expr: Expr, env: Env[Expr], default: Callable[[], Expr]) -> Expr:
        try:
            v = self._eval.visit(expr, self._cvt_env(env))
            if not expr.type_.is_scalar:
                v = tuple(v)
            return to_expr(v)
        except EvalError:
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
