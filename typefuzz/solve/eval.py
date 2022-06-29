import typing as t
from functools import reduce
from itertools import chain
from typing import Union, Iterator, Optional, Callable, cast

from numpy.random import Generator
from tvm import tir

from .store import ValueStore, ArrayNode
from ..expr.array import Tuple, List, GetItem, Len, Concat, Slice, Map, ReduceArray, ReduceRange, \
    Filter, InSet, Subset, Perm
from ..expr.basic import Env, Expr, ExprKind, Const, Var, Symbol, Range, Arith, Cmp, CmpOp, Not, \
    And, Or, ForAll, Cond, GetAttr, Dummy, to_expr
from ..expr.tensor import Num, Shape, Rank, GetDType, TensorKind, TensorDesc, LayoutMap, LayoutIndex
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
        assert type(pred) == bool
        return self.visit(cond.tr_br_, env) if pred else self.visit(cond.fls_br_, env)

    def visit_attr(self, attr: GetAttr, env: Env[ValueType]) -> ResultType:
        return self._check_not_none(self._store.query_attr(attr.name_).value, attr)

    def visit_dummy(self, dum: Dummy, env: Env[ValueType]) -> ResultType:
        raise EvalError(dum, 'Dummy expression cannot be evaluated.')

    def visit_num(self, num: Num, env: Env[ValueType]) -> ResultType:
        node = self._store.query_shape(num.t_kind_)
        return self._check_not_none(cast(ArrayNode, node).len_.value, num)

    def visit_shape(self, shape: Shape, env: Env[ValueType]) -> ResultType:
        idx = self.visit(shape.tensor_.idx_, env)
        node = self._get_tensor_shape_node(shape.tensor_.kind_, idx, shape)
        return self._check_not_none(node.value, shape)

    def visit_rank(self, rank: Rank, env: Env[ValueType]) -> ResultType:
        idx = self.visit(rank.tensor_.idx_, env)
        node = self._get_tensor_shape_node(rank.tensor_.kind_, idx, rank)
        return self._check_not_none(node.len_.value, rank)

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
        return self._check_not_none(node.value, dtype)

    def _check_not_none(self, v: ValueType, e: Expr):
        if v is None:
            raise EvalError(e, 'Result contains None.')
        if isinstance(v, (tuple, list)):
            return tuple(self._check_not_none(f, e) for f in v)
        return v

    def visit_layout_index(self, i: LayoutIndex, env: Env[ValueType]) -> ResultType:
        layout = self.visit(i.layout_, env)
        dim = self.visit(i.dim_, env)
        return int(tir.layout(layout).index_of(dim))

    def visit_layout_map(self, m: LayoutMap, env: Env[ValueType]) -> ResultType:
        tgt = self.visit(m.tgt_, env)
        src = self.visit(m.src_, env)
        layout_map = tir.bijective_layout(src, tgt)
        src_shape = tuple(self.visit(m.src_shape_, env))
        dst_shape = layout_map.forward_shape(src_shape)
        return tuple(int(d) for d in dst_shape)

    def visit_tuple(self, tup: Tuple, env: Env[ValueType]) -> ResultType:
        return (self.visit(f, env) for f in tup.fields_)

    def visit_list(self, lst: List, env: Env[ValueType]) -> ResultType:
        return (self.visit(lst.body_, env + (lst.idx_, idx))
                for idx in range(self.visit(lst.len_, env)))

    def visit_getitem(self, getitem: GetItem, env: Env[ValueType]) -> ResultType:
        arr = tuple(self.visit(getitem.arr_, env))
        idx = self.visit(getitem.idx_, env)
        if idx not in range(-len(arr), len(arr)):
            raise EvalError(getitem, 'Index out of bound.')
        return arr[idx]

    def visit_len(self, ln: Len, env: Env[ValueType]) -> ResultType:
        return len(tuple(self.visit(ln.arr_, env)))

    def visit_concat(self, concat: Concat, env: Env[ValueType]) -> ResultType:
        return chain(*(self.visit(arr, env) for arr in concat.arrays_))

    def visit_slice(self, slc: Slice, env: Env[ValueType]) -> ResultType:
        arr = tuple(self.visit(slc.arr_, env))
        begin, end = self.visit_range(slc.ran_, env)
        return arr[begin:end]

    def visit_map(self, m: Map, env: Env[ValueType]) -> ResultType:
        return map(lambda v: self.visit(m.body_, env + (m.sym_, v)), self.visit(m.arr_, env))

    def visit_reduce_array(self, red: ReduceArray, env: Env[ValueType]) -> ResultType:
        func = Arith.op_funcs[red.op_][red.type_]
        try:
            return reduce(lambda acc, v: func(acc, v), self.visit(red.arr_, env),
                          self.visit(red.init_, env))
        except TypeError as err:
            raise EvalError(red, str(err))

    def visit_reduce_index(self, red: ReduceRange, env: Env[ValueType]) -> ResultType:
        begin, end = self.visit_range(red.ran_, env)
        func = Arith.op_funcs[red.op_][red.type_]
        try:
            return reduce(lambda acc, idx: func(acc, self.visit(red.body_, env + (red.idx_, idx))),
                          range(begin, end), self.visit(red.init_, env))
        except TypeError as err:
            raise EvalError(red, str(err))

    def visit_filter(self, flt: Filter, env: Env[ValueType]) -> ResultType:
        return filter(lambda v: self.visit(flt.pred_, env + (flt.sym_, v)),
                      self.visit(flt.arr_, env))

    def visit_inset(self, inset: InSet, env: Env[ValueType]) -> ResultType:
        return self.visit(inset.elem_, env) in tuple(self.visit(inset.set_, env))

    def visit_subset(self, subset: Subset, env: Env[ValueType]) -> ResultType:
        sup = self.visit(subset.sup_, env)
        return all(map(lambda v: v in sup, self.visit(subset.sub_, env)))

    def visit_perm(self, perm: Perm, env: Env[ValueType]) -> ResultType:
        src = self.visit(perm.src_, env)
        tgt = self.visit(perm.tgt_, env)
        return set(src) == set(tgt)


class PartialEval(ExprVisitor[Env[Expr], Expr]):
    """
    Perform partial evaluation on a constraint expression to make it suitable for constraint
    solving. For some special expressions, sampling is done at this time.
    """

    def __init__(self, store: ValueStore, rng: Generator):
        super().__init__()
        self._store = store
        self._eval = EvalExpr(self._store)
        self._rng = rng

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
        v = self._store.query_var(var)
        if v is not None:
            return Const(v)
        if var.ran_ is not None:  # non-template variable must keep its original object id
            var.ran_ = self.visit_range(var.ran_, env)
        if var.choices_ is not None:
            var.choices_ = self.visit(var.choices_, env)
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
            post = self.visit(c, env)
            if post.kind == ExprKind.CONST:
                const = cast(Const, post)
                if const.val_ is False:
                    return Const(False)
                else:
                    continue
            clauses.append(post)

        return Const(True) if len(clauses) == 0 else And(*clauses)

    def visit_or(self, o: Or, env: Env[Expr]) -> Expr:
        clauses = []
        for c in o.clauses_:
            post = self.visit(c, env)
            if post.kind == ExprKind.CONST:
                const = cast(Const, post)
                if const.val_ is True:
                    return Const(True)
                else:
                    continue
            clauses.append(post)
        return Const(False) if len(clauses) == 0 else Or(*clauses)

    def visit_forall(self, forall: ForAll, env: Env[Expr]) -> Expr:
        ran = self.visit_range(forall.ran_, env)
        if ran.begin_.kind != ExprKind.CONST or ran.end_.kind != ExprKind.CONST:
            return ForAll(ran, idx=forall.idx_,
                          body=self.visit(forall.body_, env + (forall.idx_, forall.idx_)))
        begin = cast(Const, ran.begin_).val_
        end = cast(Const, ran.end_).val_
        and_expr = And(*(self.visit(forall.body_, env + (forall.idx_, Const(idx)))
                         for idx in range(begin, end)))
        return self.visit(and_expr, env)

    def visit_cond(self, cond: Cond, env: Env[Expr]) -> Expr:
        pred = self.visit(cond.pred_, env)
        if pred.kind == ExprKind.CONST:
            v = cast(Const, pred).val_
            return self.visit(cond.tr_br_, env) if v else self.visit(cond.fls_br_, env)
        else:
            return Cond(self.visit(cond.pred_, env), self.visit(cond.tr_br_, env),
                        self.visit(cond.fls_br_, env), ty=cond.type_)

    def visit_attr(self, attr: GetAttr, env: Env[Expr]) -> Expr:
        expr = self._store.query_attr(attr.name_).expr
        return attr if self._has_dummy(expr) else expr

    def _has_dummy(self, e: Expr):
        if e.kind == ExprKind.DUMMY:
            return True
        elif e.kind == ExprKind.TUPLE:
            tup = cast(Tuple, e)
            return any(self._has_dummy(f) for f in tup.fields_)
        else:
            return False

    def visit_dummy(self, dum: Dummy, env: Env[Expr]) -> Expr:
        return dum

    def visit_num(self, num: Num, env: Env[Expr]) -> Expr:
        node = self._store.query_shape(num.t_kind_)
        return cast(ArrayNode, node).len_.expr

    def visit_shape(self, shape: Shape, env: Env[Expr]) -> Expr:
        idx = self.visit(shape.index, env)
        shape = Shape(TensorDesc(shape.tensor_kind, idx))
        iv = self._try_eval(idx, env)
        if iv is None:
            return shape
        node = self._store.query_shape(shape.tensor_kind, iv)
        if node is None:
            return shape
        expr = node.expr
        if self._has_dummy(expr):
            return shape
        return expr

    def visit_rank(self, rank: Rank, env: Env[Expr]) -> Expr:
        idx = self.visit(rank.index, env)
        rank = Rank(TensorDesc(rank.tensor_kind, idx))
        iv = self._try_eval(idx, env)
        if iv is None:
            return rank
        node = self._store.query_shape(rank.tensor_.kind_, iv)
        if node is None:
            return rank
        return cast(ArrayNode, node).len_.expr

    def visit_dtype(self, dtype: GetDType, env: Env[Expr]) -> Expr:
        idx = self.visit(dtype.index, env)
        dtype = GetDType(TensorDesc(dtype.tensor_kind, idx))
        iv = self._try_eval(idx, env)
        if iv is None:
            return dtype
        node = self._store.query_dtype(dtype.tensor_.kind_, iv)
        if node is None:
            return dtype
        return node.expr

    def visit_layout_index(self, i: LayoutIndex, env: Env[Expr]) -> Expr:
        layout = self.visit(i.layout_, env)
        dim = self.visit(i.dim_, env)
        if layout.kind != ExprKind.CONST:
            return LayoutIndex(layout, dim)
        layout = cast(Const, layout)
        if dim.kind != ExprKind.CONST:
            return LayoutIndex(layout, dim)
        dim = cast(Const, dim)
        return Const(tir.layout(layout.val_).index_of(dim.val_))

    def visit_layout_map(self, m: LayoutMap, env: Env[Expr]) -> Expr:
        tgt = self.visit(m.tgt_, env)
        src = self.visit(m.src_, env)
        src_shape = self.visit(m.src_shape_, env)
        if tgt.kind != ExprKind.CONST:
            return LayoutMap(tgt, src, src_shape)
        tgt = cast(Const, tgt)
        if src.kind != ExprKind.CONST:
            return LayoutMap(tgt, src, src_shape)
        src = cast(Const, src)
        if src_shape.kind != ExprKind.TUPLE:
            return LayoutMap(tgt, src, src_shape)
        src_shape = cast(Tuple, src_shape)
        idx_map = tir.bijective_layout(src.val_, tgt.val_).forward_index(
            tuple(range(len(src_shape.fields_)))
        )
        return Tuple(*(src_shape.fields_[int(idx)] for idx in idx_map))

    def visit_tuple(self, tup: Tuple, env: Env[Expr]) -> Expr:
        return Tuple(*(self.visit(f, env) for f in tup.fields_), ty=tup.type_)

    def visit_list(self, lst: List, env: Env[Expr]) -> Expr:
        num = self._try_eval(lst.len_, env)
        if num is None:
            return lst
        return Tuple(*(self.visit(lst.body_, env + (lst.idx_, Const(idx)))
                       for idx in range(num)),
                     ty=lst.type_)

    def visit_getitem(self, getitem: GetItem, env: Env[Expr]) -> Expr:
        arr = self.visit(getitem.arr_, env)
        idx = self.visit(getitem.idx_, env)
        if arr.kind != ExprKind.TUPLE:
            return GetItem(arr, idx, ty=getitem.type_)
        arr = cast(Tuple, arr)
        if idx.kind != ExprKind.CONST:
            return GetItem(arr, idx, ty=getitem.type_)
        iv = cast(Const, idx).val_
        if iv not in range(-len(arr.fields_), len(arr.fields_)):
            return GetItem(arr, idx, ty=getitem.type_)
        return arr.fields_[iv]

    def visit_len(self, ln: Len, env: Env[Expr]) -> Expr:
        arr = self.visit(ln.arr_, env)
        if arr.kind == ExprKind.TUPLE:
            return Const(len(cast(Tuple, arr).fields_))
        elif arr.kind == ExprKind.LIST:
            return cast(List, arr).len_
        elif arr.kind == ExprKind.ATTR:
            node = self._store.query_attr(cast(GetAttr, arr).name_)
            return cast(ArrayNode, node).len_.expr
        else:
            return Len(arr)

    def visit_concat(self, concat: Concat, env: Env[Expr]) -> Expr:
        fields = []
        for i, arr in enumerate(concat.arrays_):
            arr = self.visit(arr, env)
            if arr.kind != ExprKind.TUPLE:
                return Concat(Tuple(*fields, ty=concat.type_), concat.arrays_[i:], ty=concat.type_)
            fields.extend(cast(Tuple, arr).fields_)
        return Tuple(*fields, ty=concat.type_)

    def visit_slice(self, slc: Slice, env: Env[Expr]) -> Expr:
        arr = self.visit(slc.arr_, env)
        ran = self.visit_range(slc.ran_, env)
        if arr.kind != ExprKind.TUPLE:
            return Slice(arr, ran, ty=slc.type_)
        ran_val = self._try_eval(ran, env)
        if ran_val is None:
            return Slice(arr, ran, ty=slc.type_)
        return Tuple(*cast(Tuple, arr).fields_[ran_val[0]:ran_val[1]], ty=slc.type_)

    def visit_map(self, m: Map, env: Env[Expr]) -> Expr:
        arr = self.visit(m.arr_, env)
        if arr.kind != ExprKind.TUPLE:
            return Map(arr, sym=m.sym_, body=self.visit(m.body_, env + (m.sym_, m.sym_)),
                       ty=m.type_)
        arr = cast(Tuple, arr)
        return Tuple(*(self.visit(m.body_, env + (m.sym_, e)) for e in arr.fields_), ty=m.type_)

    def visit_reduce_array(self, red: ReduceArray, env: Env[Expr]) -> Expr:
        return self._try_fold(
            red, env, lambda: ReduceArray(
                self.visit(red.arr_, env), red.op_, self.visit(red.init_, env), ty=red.type_
            )
        )

    def visit_reduce_index(self, red: ReduceRange, env: Env[Expr]) -> Expr:
        return self._try_fold(
            red, env, lambda: ReduceRange(
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
        # Try folding expression
        inset = self._try_fold(
            inset, env, lambda: InSet(
                self.visit(inset.elem_, env), self.visit(inset.set_, env)
            )
        )
        if inset.kind == ExprKind.CONST:
            return inset

        # Check if the expression can be sampled
        inset = cast(InSet, inset)
        if inset.set_.kind != ExprKind.TUPLE:
            return inset
        tup = cast(Tuple, inset.set_)

        # Sample one element from tuple
        idx = self._rng.choice(len(tup.fields_))
        return Cmp(CmpOp.EQ, inset.elem_, tup.fields_[idx])

    def visit_subset(self, subset: Subset, env: Env[Expr]) -> Expr:
        # Try folding expression
        subset = self._try_fold(
            subset, env, lambda: Subset(
                self.visit(subset.sub_, env), self.visit(subset.sup_, env)
            )
        )
        if subset.kind == ExprKind.CONST:
            return subset

        # Check if the expression can be sampled
        subset = cast(Subset, subset)
        if subset.sup_.kind != ExprKind.TUPLE:
            return subset
        sup = cast(Tuple, subset.sup_)

        # Sample each element in superset to create subset
        sub = subset.sub_
        sub_sp = [e for e in sup.fields_ if self._rng.choice(2)]
        return And(
            Cmp(CmpOp.EQ, Len(sub), len(sub_sp)),
            *(Cmp(CmpOp.EQ, GetItem(sub, i, ty=sub.type_.elem_type), e)
              for i, e in enumerate(sub_sp))
        )

    def visit_perm(self, perm: Perm, env: Env[Expr]) -> Expr:
        # Try folding expression
        perm = self._try_fold(
            perm, env, lambda: Perm(self.visit(perm.tgt_, env), self.visit(perm.src_, env))
        )
        if perm.kind == ExprKind.CONST:
            return perm

        # Check if the expression can be sampled
        perm = cast(Perm, perm)
        if perm.src_.kind != ExprKind.TUPLE:
            return perm
        src = cast(Tuple, perm.src_)

        # Permute source array
        perm_ind = self._rng.permutation(range(len(src.fields_)))
        tgt = perm.tgt_
        return And(
            Cmp(CmpOp.EQ, Len(perm.tgt_), len(src.fields_)),
            *(Cmp(CmpOp.EQ, GetItem(perm.tgt_, i, ty=tgt.type_.elem_type), src.fields_[pi])
              for i, pi in enumerate(perm_ind))
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
            val_env += sym, const.val_
        return val_env
