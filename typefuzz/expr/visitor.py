import typing as t
from typing import Generic, TypeVar, Dict, Callable, Any, Iterable, Optional, cast

from .array import Tuple, List, GetItem, Len, Concat, Slice, Map, ReduceArray, ReduceRange, \
    Filter, InSet, Subset, Perm
from .basic import Expr, ExprKind, Const, Var, Symbol, Range, Arith, Cmp, Not, And, Or, ForAll, \
    Cond, GetAttr, Dummy, Env
from .tensor import Num, Shape, Rank, GetDType, TensorDesc, LayoutIndex, LayoutMap
from .ty import Type, TypeKind, BoolType, IntType, FloatType, StrType, DType, TupleType, ListType, \
    TyVar
from ..util import map_opt

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
            ExprKind.DUMMY: self.visit_dummy,
            ExprKind.NUM: self.visit_num,
            ExprKind.SHAPE: self.visit_shape,
            ExprKind.RANK: self.visit_rank,
            ExprKind.DTYPE: self.visit_dtype,
            ExprKind.LAYOUT_INDEX: self.visit_layout_index,
            ExprKind.LAYOUT_MAP: self.visit_layout_map,
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
            ExprKind.PERM: self.visit_perm,
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

    def visit_dummy(self, dum: Dummy, arg: A) -> R:
        pass

    def visit_num(self, num: Num, arg: A) -> R:
        return self._visit_sub(num, arg)

    def visit_shape(self, shape: Shape, arg: A) -> R:
        return self._visit_sub(shape, arg)

    def visit_rank(self, rank: Rank, arg: A) -> R:
        return self._visit_sub(rank, arg)

    def visit_dtype(self, dtype: GetDType, arg: A) -> R:
        return self._visit_sub(dtype, arg)

    def visit_layout_index(self, i: LayoutIndex, arg: A) -> R:
        return self._visit_sub(i, arg)

    def visit_layout_map(self, m: LayoutMap, arg: A) -> R:
        return self._visit_sub(m, arg)

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

    def visit_reduce_index(self, red: ReduceRange, arg: A) -> R:
        return self._visit_sub(red, arg)

    def visit_filter(self, flt: Filter, arg: A) -> R:
        return self._visit_sub(flt, arg)

    def visit_inset(self, inset: InSet, arg: A) -> R:
        return self._visit_sub(inset, arg)

    def visit_subset(self, subset: Subset, arg: A) -> R:
        return self._visit_sub(subset, arg)

    def visit_perm(self, perm: Perm, arg: A) -> R:
        return self._visit_sub(perm, arg)

    def _visit_sub(self, expr: Expr, arg: A):
        for s in expr.sub_expr_:
            self.visit(s, arg)


class TypeVisitor(Generic[A, R]):
    """
    Base class for type visitors.
    """

    def __init__(self):
        self._methods: Dict[TypeKind, Callable[[Any, A], R]] = {
            TypeKind.bool: self.visit_bool,
            TypeKind.int: self.visit_int,
            TypeKind.float: self.visit_float,
            TypeKind.str: self.visit_str,
            TypeKind.dtype: self.visit_dtype,
            TypeKind.tuple: self.visit_tuple,
            TypeKind.list: self.visit_list,
            TypeKind.var: self.visit_var,
        }

    def visit(self, ty: Type, arg: A) -> R:
        return self._methods[ty.kind](ty, arg)

    def visit_bool(self, b: BoolType, arg: A) -> R:
        pass

    def visit_int(self, i: IntType, arg: A) -> R:
        pass

    def visit_float(self, f: FloatType, arg: A) -> R:
        pass

    def visit_str(self, s: StrType, arg: A) -> R:
        pass

    def visit_dtype(self, dtype: DType, arg: A) -> R:
        pass

    def visit_tuple(self, tup: TupleType, arg: A) -> R:
        for ty in tup.field_ty_:
            self.visit(ty, arg)
        return None

    def visit_list(self, lst: ListType, arg: A) -> R:
        self.visit(lst.elem_ty_, arg)
        return None

    def visit_var(self, var: TyVar, arg: A) -> R:
        pass


class StructuralEq(ExprVisitor[Expr, bool]):
    """
    Whether two expressions are structurally equal.
    """

    def visit(self, this: Expr, other: Expr) -> bool:
        if this.kind != other.kind:
            return False
        return super().visit(this, other)

    def visit_const(self, const: Const, other: Expr) -> bool:
        other = cast(Const, other)
        return const.val_ == other.val_

    def visit_var(self, var: Var, other: Expr) -> bool:
        other = cast(Var, other)
        if var.tmpl_ != other.tmpl_:
            return False
        return self._cmp_opt([(var.ran_, other.ran_), (var.choices_, other.choices_)])

    def visit_symbol(self, sym: Symbol, other: Expr) -> bool:
        return sym is other

    def visit_range(self, ran: Range, other: Expr) -> bool:
        other = cast(Range, other)
        return self._cmp_opt([(ran.begin_, other.begin_), (ran.end_, other.end_)])

    def visit_arith(self, arith: Arith, other: Expr) -> bool:
        other = cast(Arith, other)
        if arith.op_ != other.op_:
            return False
        return self._cmp_expr([(arith.lhs_, other.lhs_), (arith.rhs_, other.rhs_)])

    def visit_cmp(self, cmp: Cmp, other: Expr) -> bool:
        other = cast(Cmp, other)
        if cmp.op_ != other.op_:
            return False
        return self._cmp_expr([(cmp.lhs_, other.lhs_), (cmp.rhs_, other.rhs_)])

    def visit_not(self, n: Not, other: Expr) -> bool:
        other = cast(Not, other)
        return self._cmp_expr([(n.prop_, other.prop_)])

    def visit_and(self, a: And, other: Expr) -> bool:
        other = cast(And, a)
        return self._cmp_list(a.clauses_, other.clauses_)

    def visit_or(self, o: Or, other: Expr) -> bool:
        other = cast(Or, o)
        return self._cmp_list(o.clauses_, other.clauses_)

    def visit_forall(self, forall: ForAll, other: Expr) -> bool:
        other = cast(ForAll, other)
        return self._cmp_expr([(forall.ran_, other.ran_), (forall.body_, other.body_)])

    def visit_cond(self, cond: Cond, other: Expr) -> bool:
        other = cast(Cond, cond)
        return self._cmp_expr([(cond.pred_, other.pred_), (cond.tr_br_, other.tr_br_),
                               (cond.fls_br_, other.fls_br_)])

    def visit_attr(self, attr: GetAttr, other: Expr) -> bool:
        other = cast(GetAttr, other)
        return attr.name_ == other.name_

    def visit_dummy(self, dum: Dummy, other: Expr) -> bool:
        return True

    def visit_num(self, num: Num, other: Expr) -> bool:
        other = cast(Num, num)
        return num.t_kind_ == other.t_kind_

    def visit_shape(self, shape: Shape, other: Expr) -> bool:
        other = cast(Shape, other)
        return self._cmp_tensor(shape.tensor_, other.tensor_)

    def visit_rank(self, rank: Rank, other: Expr) -> bool:
        other = cast(Rank, other)
        return self._cmp_tensor(rank.tensor_, other.tensor_)

    def visit_dtype(self, dtype: GetDType, other: Expr) -> bool:
        other = cast(GetDType, other)
        return self._cmp_tensor(dtype.tensor_, other.tensor_)

    def visit_layout_index(self, i: LayoutIndex, other: Expr) -> bool:
        other = cast(LayoutIndex, other)
        return self._cmp_expr([(i.layout_, other.layout_), (i.dim_, other.dim_)])

    def visit_layout_map(self, m: LayoutMap, other: Expr) -> bool:
        other = cast(LayoutMap, other)
        return self._cmp_expr([(m.tgt_, other.tgt_), (m.src_, other.src_),
                               (m.src_shape_, other.src_shape_)])

    def visit_tuple(self, tup: Tuple, other: Expr) -> bool:
        other = cast(Tuple, tup)
        return self._cmp_list(tup.fields_, other.fields_)

    def visit_list(self, lst: List, other: Expr) -> bool:
        other = cast(List, lst)
        return self._cmp_expr([(lst.len_, other.len_), (lst.body_, other.type_)])

    def visit_getitem(self, getitem: GetItem, other: Expr) -> bool:
        other = cast(GetItem, other)
        return self._cmp_expr([(getitem.arr_, other.arr_), (getitem.idx_, other.idx_)])

    def visit_len(self, ln: Len, other: Expr) -> bool:
        other = cast(Len, ln)
        return self._cmp_expr([(ln.arr_, other.arr_)])

    def visit_concat(self, concat: Concat, other: Expr) -> bool:
        other = cast(Concat, concat)
        return self._cmp_list(concat.arrays_, other.arrays_)

    def visit_slice(self, slc: Slice, other: Expr) -> bool:
        other = cast(Slice, other)
        return self._cmp_expr([(slc.arr_, other.arr_), (slc.ran_, other.ran_)])

    def visit_map(self, m: Map, other: Expr) -> bool:
        other = cast(Map, other)
        return self._cmp_expr([(m.arr_, other.arr_), (m.body_, other.body_)])

    def visit_reduce_array(self, red: ReduceArray, other: Expr) -> bool:
        other = cast(ReduceArray, other)
        if red.op_ != other.op_:
            return False
        return self._cmp_expr([(red.arr_, other.arr_), (red.init_, other.init_)])

    def visit_reduce_index(self, red: ReduceRange, other: Expr) -> bool:
        other = cast(ReduceRange, other)
        if red.op_ != other.op_:
            return False
        return self._cmp_expr([(red.ran_, other.ran_), (red.body_, other.body_),
                               (red.init_, other.init_)])

    def visit_filter(self, flt: Filter, other: Expr) -> bool:
        other = cast(Filter, flt)
        return self._cmp_expr([(flt.arr_, other.arr_), (flt.pred_, other.pred_)])

    def visit_inset(self, inset: InSet, other: Expr) -> bool:
        other = cast(InSet, other)
        return self._cmp_expr([(inset.elem_, other.elem_), (inset.set_, other.set_)])

    def visit_subset(self, subset: Subset, other: Expr) -> bool:
        other = cast(Subset, subset)
        return self._cmp_expr([(subset.sub_, other.sub_), (subset.sup_, other.sup_)])

    def visit_perm(self, perm: Perm, other: Expr) -> bool:
        other = cast(Perm, other)
        return self._cmp_expr([(perm.tgt_, other.tgt_), (perm.src_, other.src_)])

    def _cmp_opt(self, pairs: Iterable[t.Tuple[Optional[Expr], Optional[Expr]]]):
        real_pairs: t.List[t.Tuple[Expr, Expr]] = []
        for this, other in pairs:
            if (this is not None and other is None) or (this is None and other is not None):
                return False
            if (this is not None) and (other is not None):
                real_pairs.append((this, other))
        return self._cmp_expr(real_pairs)

    def _cmp_list(self, this: t.List[Expr], other: t.List[Expr]):
        if len(this) != len(other):
            return False
        return self._cmp_expr(zip(this, other))

    def _cmp_expr(self, pairs: Iterable[t.Tuple[Expr, Expr]]):
        return all(map(lambda p: self.visit(p[0], p[1]), pairs))

    def _cmp_tensor(self, this: TensorDesc, other: TensorDesc):
        return this.kind_ == other.kind_ and self.visit(this.idx_, other.idx_)


class CopyExpr(ExprVisitor[Env[Symbol], Expr]):
    def copy(self, expr: Expr):
        return self.visit(expr, Env())

    def visit_const(self, const: Const, env: Env[Symbol]) -> Expr:
        return Const(const.val_)

    def visit_var(self, var: Var, env: Env[Symbol]) -> Expr:
        return Var(ty=var.type_, ran=map_opt(lambda ran: self.visit(ran, env), var.ran_),
                   choices=map_opt(lambda c: self.visit(c, env), var.choices_), tmpl=var.tmpl_)

    def visit_symbol(self, sym: Symbol, env: Env[Symbol]) -> Expr:
        return env[sym]

    def visit_range(self, ran: Range, env: Env[Symbol]) -> Range:
        return Range(begin=map_opt(lambda beg: self.visit(beg, env), ran.begin_),
                     end=map_opt(lambda end: self.visit(end, env), ran.end_),
                     ty=ran.type_)

    def visit_arith(self, arith: Arith, env: Env[Symbol]) -> Expr:
        return Arith(arith.op_, self.visit(arith.lhs_, env), self.visit(arith.rhs_, env),
                     ty=arith.type_)

    def visit_cmp(self, cmp: Cmp, env: Env[Symbol]) -> Expr:
        return Cmp(cmp.op_, self.visit(cmp.lhs_, env), self.visit(cmp.rhs_, env))

    def visit_not(self, n: Not, env: Env[Symbol]) -> Expr:
        return Not(self.visit(n.prop_, env))

    def visit_and(self, a: And, env: Env[Symbol]) -> Expr:
        return And(*(self.visit(c, env) for c in a.clauses_))

    def visit_or(self, o: Or, env: Env[Symbol]) -> Expr:
        return Or(*(self.visit(c, env) for c in o.clauses_))

    def visit_forall(self, forall: ForAll, env: Env[Symbol]) -> Expr:
        idx = Symbol(ty=forall.idx_.type_)
        return ForAll(ran=self.visit_range(forall.ran_, env), idx=idx,
                      body=self.visit(forall.body_, env + (forall.idx_, idx)))

    def visit_cond(self, cond: Cond, env: Env[Symbol]) -> Expr:
        return Cond(self.visit(cond.pred_, env), self.visit(cond.tr_br_, env),
                    self.visit(cond.fls_br_, env), ty=cond.type_)

    def visit_attr(self, attr: GetAttr, env: Env[Symbol]) -> Expr:
        return GetAttr(attr.name_, ty=attr.type_)

    def visit_dummy(self, dum: Dummy, env: Env[Symbol]) -> Expr:
        return Dummy()

    def visit_num(self, num: Num, env: Env[Symbol]) -> Expr:
        return Num(num.t_kind_)

    def visit_shape(self, shape: Shape, env: Env[Symbol]) -> Expr:
        return Shape(self._cp_tensor(shape.tensor_, env))

    def visit_rank(self, rank: Rank, env: Env[Symbol]) -> Expr:
        return Rank(self._cp_tensor(rank.tensor_, env))

    def visit_dtype(self, dtype: GetDType, env: Env[Symbol]) -> Expr:
        return GetDType(self._cp_tensor(dtype.tensor_, env))

    def visit_layout_index(self, i: LayoutIndex, env: Env[Symbol]) -> Expr:
        return LayoutIndex(self.visit(i.layout_, env), self.visit(i.dim_, env))

    def visit_layout_map(self, m: LayoutMap, env: Env[Symbol]) -> Expr:
        return LayoutMap(self.visit(m.tgt_, env), self.visit(m.src_, env),
                         self.visit(m.src_shape_, env))

    def _cp_tensor(self, tensor: TensorDesc, env: Env[Symbol]):
        return TensorDesc(tensor.kind_, self.visit(tensor.idx_, env))

    def visit_tuple(self, tup: Tuple, env: Env[Symbol]) -> Expr:
        return Tuple(*(self.visit(e, env) for e in tup.fields_), ty=tup.type_)

    def visit_list(self, lst: List, env: Env[Symbol]) -> Expr:
        idx = Symbol(ty=lst.idx_.type_)
        return List(self.visit(lst.len_, env), idx=idx,
                    body=self.visit(lst.body_, env + (lst.idx_, idx)), ty=lst.type_)

    def visit_getitem(self, getitem: GetItem, env: Env[Symbol]) -> Expr:
        return GetItem(self.visit(getitem.arr_, env), self.visit(getitem.idx_, env),
                       ty=getitem.type_)

    def visit_len(self, ln: Len, env: Env[Symbol]) -> Expr:
        return Len(self.visit(ln.arr_, env))

    def visit_concat(self, concat: Concat, env: Env[Symbol]) -> Expr:
        return Concat(*(self.visit(arr, env) for arr in concat.arrays_), ty=concat.type_)

    def visit_slice(self, slc: Slice, env: Env[Symbol]) -> Expr:
        return Slice(self.visit(slc.arr_, env), self.visit_range(slc.ran_, env), ty=slc.type_)

    def visit_map(self, m: Map, env: Env[Symbol]) -> Expr:
        sym = Symbol(ty=m.sym_.type_)
        return Map(self.visit(m.arr_, env), sym=sym, body=self.visit(m.body_, env + (m.sym_, sym)),
                   ty=m.type_)

    def visit_reduce_array(self, red: ReduceArray, env: Env[Symbol]) -> Expr:
        return ReduceArray(self.visit(red.arr_, env), red.op_, self.visit(red.init_, env),
                           ty=red.type_)

    def visit_reduce_index(self, red: ReduceRange, env: Env[Symbol]) -> Expr:
        idx = Symbol(ty=red.idx_.type_)
        return ReduceRange(
            self.visit_range(red.ran_, env), red.op_, self.visit(red.init_, env), idx=idx,
            body=self.visit(red.body_, env + (red.idx_, idx)), ty=red.type_
        )

    def visit_filter(self, flt: Filter, env: Env[Symbol]) -> Expr:
        sym = Symbol()
        return Filter(self.visit(flt.arr_, env), sym=sym,
                      pred=self.visit(flt.pred_, env + (flt.sym_, sym)), ty=flt.type_)

    def visit_inset(self, inset: InSet, env: Env[Symbol]) -> Expr:
        return InSet(self.visit(inset.elem_, env), self.visit(inset.set_, env))

    def visit_subset(self, subset: Subset, env: Env[Symbol]) -> Expr:
        return Subset(self.visit(subset.sub_, env), self.visit(subset.sup_, env))

    def visit_perm(self, perm: Perm, env: Env[Symbol]) -> Expr:
        return Perm(self.visit(perm.tgt_, env), self.visit(perm.src_, env))
