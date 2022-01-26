import typing as t
from functools import reduce
from itertools import chain, islice
from typing import Union, Iterator, cast

from .store import ValueStore, ArrayNode
from ..expr import ExprVisitor
from ..expr.array import Tuple, List, GetItem, Len, Concat, Slice, Map, ReduceArray, ReduceIndex, \
    Filter, InSet, Subset
from ..expr.basic import Env, Expr, Const, Var, Symbol, Range, Arith, Cmp, Not, And, Or, ForAll, \
    Cond, GetAttr
from ..expr.tensor import Num, Shape, Rank, GetDType, TensorKind
from ..expr.ty import ValueType
from ..util import map_opt


class EvalFailed(Exception):
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

    def visit(self, e: Expr, env: Env[ValueType]) -> ResultType:
        v = self._methods[e.kind](e, env)
        if v is None:
            # Only query from value store will produce None.
            raise EvalFailed(e, 'Cannot query from value store.')
        return v

    def visit_const(self, const: Const, env: Env[ValueType]) -> ResultType:
        return const.val_

    def visit_var(self, var: Var, env: Env[ValueType]) -> ResultType:
        raise EvalFailed(var, 'Variable cannot be evaluated.')

    def visit_symbol(self, sym: Symbol, env: Env[ValueType]) -> ResultType:
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

    def visit_num(self, num: Num, env: Env[ValueType]) -> ResultType:
        if num.t_kind_ == TensorKind.input:
            node = self._store.in_shapes_
        else:
            node = self._store.out_shapes_
        return node.len_.value

    def visit_shape(self, shape: Shape, env: Env[ValueType]) -> ResultType:
        idx = self.visit(shape.tensor_.idx_, env)
        node = self._get_tensor_shape_node(shape.tensor_.kind_, idx, shape)
        return node.value

    def visit_rank(self, rank: Rank, env: Env[ValueType]) -> ResultType:
        idx = self.visit(rank.tensor_.idx_, env)
        node = self._get_tensor_shape_node(rank.tensor_.kind_, idx, rank)
        return cast(ArrayNode, node).len_.value

    def _get_tensor_shape_node(self, kind: TensorKind, idx: int, e: Expr):
        if kind == TensorKind.input:
            node = self._store.query_in_shape(idx)
        else:
            node = self._store.query_out_shape(idx)
        if node is None:
            raise EvalFailed(
                e, f'Shape of {kind.name} tensor {idx} is undefined.'
            )
        return node

    def visit_dtype(self, dtype: GetDType, env: Env[ValueType]) -> ResultType:
        kind = dtype.tensor_.kind_
        idx = self.visit(dtype.tensor_.idx_, env)
        if kind == TensorKind.input:
            node = self._store.query_in_dtype(idx)
        else:
            node = self._store.query_out_dtype(idx)
        if node is None:
            raise EvalFailed(
                dtype, f'Data type of {kind.name} tensor {idx} is undefined.'
            )
        return node

    def visit_tuple(self, tup: Tuple, env: Env[ValueType]) -> ResultType:
        return (self.visit(f, env) for f in tup.fields_)

    def visit_list(self, lst: List, env: Env[ValueType]) -> ResultType:
        return (self.visit(lst.body_, env + (lst.idx_, idx))
                for idx in range(self.visit(lst.len_, env)))

    def visit_getitem(self, getitem: GetItem, env: Env[ValueType]) -> ResultType:
        arr = tuple(self.visit(getitem.arr_, env))
        idx = self.visit(getitem.idx_, env)
        if idx >= len(arr):
            raise EvalFailed(getitem, 'Index out of bound.')
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
