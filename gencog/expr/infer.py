import typing as t
from functools import reduce
from typing import NamedTuple, Dict, cast

from .array import Tuple, List, GetItem, Len, Concat, Slice, Map, ReduceArray, ReduceRange, \
    Filter, InSet, Subset, Perm
from .basic import Expr, ExprKind, Env, Const, Var, Symbol, Range, Arith, ArithOp, Cmp, Not, And, \
    Or, ForAll, Cond, GetAttr, Dummy
from .tensor import Num, Shape, Rank, GetDType, LayoutMap, LayoutIndex
from .ty import Type, TypeKind, BoolType, IntType, FloatType, StrType, DType, TupleType, ListType, \
    TyVar, BOOL, INT, STR, DTYPE
from .visitor import TypeVisitor, ExprVisitor
from ..util import unwrap, unwrap_or, filter_none, cls_name


class ExprTypeError(Exception):
    def __init__(self, expr: Expr, msg: str):
        self.expr_ = expr
        self.msg_ = msg


def infer_type(e: Expr, attr_ty: Dict[str, Type], hint: Type = TyVar()):
    return TypeInfer(attr_ty).visit(e, InferArg(Env(), hint))


class InferArg(NamedTuple):
    env: Env[Type]
    hint: Type


class TypeInfer(ExprVisitor[InferArg, Type]):
    """
    Infer and check types of constraint expressions.
    """

    def __init__(self, attr_ty: Dict[str, Type]):
        super().__init__()
        self._attr_ty = attr_ty
        self._unifier = TypeUnifier()
        self._var_chk = TypeVarChecker()

    def visit(self, e: Expr, arg: InferArg) -> Type:
        try:
            ty = super().visit(e, arg)
        except UnificationError as err:
            raise ExprTypeError(e, str(err))
        if self._var_chk.visit(ty, None):
            raise ExprTypeError(e, f'Cannot infer type for {cls_name(e)}.')
        if e.type_ is not None:
            try:
                self._unify(e.type_, ty)
            except UnificationError:
                raise ExprTypeError(
                    e, f'Inferred type {ty} is not consistent with annotated type {e.type_}.'
                )
        e.type_ = ty
        return ty

    def visit_const(self, const: Const, arg: InferArg) -> Type:
        return self._unify(const.type_, arg.hint)

    def visit_var(self, var: Var, arg: InferArg) -> Type:
        ty = self._unify(unwrap_or(var.type_, TyVar()), arg.hint)
        if var.ran_ is not None:
            ty = self.visit(var.ran_, InferArg(arg.env, ty))
        if var.choices_ is not None:
            self.visit(var.choices_, InferArg(arg.env, ListType(ty)))
        return ty

    def visit_symbol(self, sym: Symbol, arg: InferArg) -> Type:
        return arg.env[sym]

    def visit_range(self, ran: Range, arg: InferArg) -> Type:
        ty = self._unify_expr(filter_none([ran.begin_, ran.end_]), arg)
        if ty.kind not in ran.valid_type_kinds:
            raise ExprTypeError(ran, f'Range not supported for type {ty}.')
        return ty

    def visit_arith(self, arith: Arith, arg: InferArg) -> Type:
        ty = self._unify_expr([arith.lhs_, arith.rhs_], arg)
        return self._check_arith_type(arith, arith.op_, ty)

    @staticmethod
    def _check_arith_type(e: Expr, op: ArithOp, ty: Type):
        if ty not in Arith.op_funcs[op]:
            raise ExprTypeError(
                e, f'Arithmetic operator {op.value} not supported for type {ty}.'
            )
        return ty

    def visit_cmp(self, cmp: Cmp, arg: InferArg) -> Type:
        ty = self._unify_expr([cmp.lhs_, cmp.rhs_], InferArg(arg.env, TyVar()))
        if ty not in Cmp.op_funcs[cmp.op_]:
            raise ExprTypeError(
                cmp, f'Comparison operator {cmp.op_.value} not supported for type {ty}.'
            )
        return self._unify(arg.hint, BOOL)

    def visit_not(self, n: Not, arg: InferArg) -> Type:
        self.visit(n.prop_, InferArg(arg.env, BOOL))
        return self._unify(arg.hint, BOOL)

    def visit_and(self, a: And, arg: InferArg) -> Type:
        self._unify_expr(a.clauses_, InferArg(arg.env, BOOL))
        return self._unify(arg.hint, BOOL)

    def visit_or(self, o: Or, arg: InferArg) -> Type:
        self._unify_expr(o.clauses_, InferArg(arg.env, BOOL))
        return self._unify(arg.hint, BOOL)

    def visit_forall(self, forall: ForAll, arg: InferArg) -> Type:
        self.visit(forall.ran_, InferArg(arg.env, INT))
        env = arg.env + (forall.idx_, INT)
        return self.visit(forall.body_, InferArg(env, arg.hint))

    def visit_cond(self, cond: Cond, arg: InferArg) -> Type:
        self.visit(cond.pred_, InferArg(arg.env, BOOL))
        return self._unify_expr([cond.tr_br_, cond.fls_br_], arg)

    def visit_attr(self, attr: GetAttr, arg: InferArg) -> Type:
        if attr.name_ not in self._attr_ty:
            raise ExprTypeError(
                attr, f'Attribute \'{attr.name_}\' undefined.'
            )
        return self._attr_ty[attr.name_]

    def visit_dummy(self, dum: Dummy, arg: InferArg) -> Type:
        raise ExprTypeError(dum, 'Dummy expression cannot appear in user code.')

    def visit_num(self, num: Num, arg: InferArg) -> Type:
        return self._unify(arg.hint, INT)

    def visit_shape(self, shape: Shape, arg: InferArg) -> Type:
        return self._unify(arg.hint, ListType(INT))

    def visit_rank(self, rank: Rank, arg: InferArg) -> Type:
        return self._unify(arg.hint, INT)

    def visit_dtype(self, dtype: GetDType, arg: InferArg) -> Type:
        return self._unify(arg.hint, DTYPE)

    def visit_layout_index(self, i: LayoutIndex, arg: InferArg) -> Type:
        self.visit(i.layout_, InferArg(arg.env, STR))
        self.visit(i.dim_, InferArg(arg.env, STR))
        return self._unify(arg.hint, INT)

    def visit_layout_map(self, m: LayoutMap, arg: InferArg) -> Type:
        self.visit(m.tgt_, InferArg(arg.env, STR))
        self.visit(m.src_, InferArg(arg.env, STR))
        self.visit(m.src_shape_, InferArg(arg.env, ListType(INT)))
        return self._unify(arg.hint, ListType(INT))

    def visit_tuple(self, tup: Tuple, arg: InferArg) -> Type:
        hint = arg.hint
        tup_len = len(tup.fields_)
        if tup_len == 0:  # no information provided by tuple fields
            return arg.hint  # use hint directly
        if hint.kind == TypeKind.tuple:
            field_hint = cast(TupleType, hint).field_ty_
        elif hint.kind == TypeKind.list:
            field_hint = [cast(ListType, hint).elem_ty_ for _ in range(tup_len)]
        elif hint.kind == TypeKind.var:
            field_hint = [TyVar() for _ in range(tup_len)]
        else:
            raise ExprTypeError(
                tup, f'Incompatible type {hint} for {cls_name(tup)}.'
            )
        field_ty = (self.visit(e, InferArg(arg.env, hint)) for e, hint in
                    zip(tup.fields_, field_hint))
        return TupleType(*field_ty)

    def visit_list(self, lst: List, arg: InferArg) -> Type:
        self.visit(lst.len_, InferArg(arg.env, INT))
        env = arg.env + (lst.idx_, INT)
        elem_hint = self._create_elem_hint(lst, arg.hint)
        elem_ty = self.visit(lst.body_, InferArg(env, elem_hint))
        return ListType(elem_ty)

    def visit_getitem(self, getitem: GetItem, arg: InferArg) -> Type:
        arr_ty = self.visit(getitem.arr_, InferArg(arg.env, TyVar()))
        self.visit(getitem.idx_, InferArg(arg.env, INT))
        if arr_ty.kind == TypeKind.tuple:
            arr_ty = cast(TupleType, arr_ty)
            if arr_ty.is_homo_:
                return self._unify(arr_ty.elem_type, arg.hint)
            elif getitem.idx_.kind == ExprKind.CONST:
                return self._unify(arr_ty.field_ty_[cast(Const, getitem.idx_).val_], arg.hint)
            else:
                raise ExprTypeError(
                    getitem,
                    f'Cannot infer type for {cls_name(getitem)} that gets non-constant item '
                    f'from heterogeneous tuple.'
                )
        elif arr_ty.kind == TypeKind.list:
            return self._unify(cast(ListType, arr_ty).elem_ty_, arg.hint)
        else:
            raise ExprTypeError(
                getitem, f'Cannot get item from type {arr_ty}.'
            )

    def visit_len(self, ln: Len, arg: InferArg) -> Type:
        self.visit(ln.arr_, InferArg(arg.env, ListType(TyVar())))
        return self._unify(arg.hint, INT)

    def visit_concat(self, concat: Concat, arg: InferArg) -> Type:
        elem_hint = self._create_elem_hint(concat, arg.hint)
        return self._unify_expr(concat.arrays_, InferArg(arg.env, ListType(elem_hint)))

    def visit_slice(self, slc: Slice, arg: InferArg) -> Type:
        elem_hint = self._create_elem_hint(slc, arg.hint)
        self.visit(slc.ran_, InferArg(arg.env, INT))
        return self.visit(slc.arr_, InferArg(arg.env, ListType(elem_hint)))

    def visit_map(self, m: Map, arg: InferArg) -> Type:
        arr_ty = self.visit(m.arr_, InferArg(arg.env, ListType(TyVar())))
        elem_ty = self._get_elem_type(m.arr_, arr_ty)
        env = arg.env + (m.sym_, elem_ty)
        mapped_elem_hint = self._create_elem_hint(m, arg.hint)
        return ListType(self.visit(m.body_, InferArg(env, mapped_elem_hint)))

    def visit_reduce_array(self, red: ReduceArray, arg: InferArg) -> Type:
        arr_ty = self.visit(red.arr_, InferArg(arg.env, ListType(arg.hint)))
        elem_ty = self._get_elem_type(red, arr_ty)
        ty = self.visit(red.init_, InferArg(arg.env, elem_ty))
        return self._check_arith_type(red, red.op_, ty)

    def visit_reduce_index(self, red: ReduceRange, arg: InferArg) -> Type:
        self.visit(red.ran_, InferArg(arg.env, INT))
        env = arg.env + (red.idx_, INT)
        ty = self._unify_expr([red.body_, red.init_], InferArg(env, arg.hint))
        return self._check_arith_type(red, red.op_, ty)

    def visit_filter(self, flt: Filter, arg: InferArg) -> Type:
        elem_hint = self._create_elem_hint(flt, arg.hint)
        arr_ty = self.visit(flt.arr_, InferArg(arg.env, ListType(elem_hint)))
        elem_ty = self._get_elem_type(flt, arr_ty)
        env = arg.env + (flt.sym_, elem_ty)
        self.visit(flt.pred_, InferArg(env, BOOL))
        return ListType(elem_ty)

    def visit_inset(self, inset: InSet, arg: InferArg) -> Type:
        elem_ty = self.visit(inset.elem_, InferArg(arg.env, TyVar()))
        self.visit(inset.set_, InferArg(arg.env, ListType(elem_ty)))
        return self._unify(arg.hint, BOOL)

    def visit_subset(self, subset: Subset, arg: InferArg) -> Type:
        sub_ty = self.visit(subset.sub_, InferArg(arg.env, ListType(TyVar())))
        elem_ty = self._get_elem_type(subset, sub_ty)
        self.visit(subset.sup_, InferArg(arg.env, ListType(elem_ty)))
        return self._unify(arg.hint, BOOL)

    def visit_perm(self, perm: Perm, arg: InferArg) -> Type:
        src_ty = self.visit(perm.src_, InferArg(arg.env, ListType(TyVar())))
        elem_ty = self._get_elem_type(perm, src_ty)
        self.visit(perm.tgt_, InferArg(arg.env, ListType(elem_ty)))
        return self._unify(arg.hint, BOOL)

    def _unify(self, lhs: Type, rhs: Type):
        return self._unifier.visit(lhs, rhs)

    def _unify_expr(self, expr_list: t.List[Expr], arg: InferArg):
        return reduce(lambda ty, expr: self._unify(ty, self.visit(expr, InferArg(arg.env, ty))),
                      expr_list, arg.hint)

    @staticmethod
    def _create_elem_hint(e: Expr, hint: Type):
        if hint.kind == TypeKind.list:
            elem_hint = cast(ListType, hint).elem_ty_
        elif hint.kind == TypeKind.var:
            elem_hint = TyVar()
        else:
            raise ExprTypeError(
                e, f'Incompatible type {hint} for {cls_name(e)}.'
            )
        return elem_hint

    @staticmethod
    def _get_elem_type(e: Expr, ty: Type):
        elem_ty = ty.elem_type
        if ty.elem_type is not None:
            return unwrap(elem_ty)
        else:
            raise ExprTypeError(
                e, f'Cannot get element type for {ty}.'
            )


class TypeVarChecker(TypeVisitor[None, bool]):
    def visit_bool(self, b: BoolType, arg: None) -> bool:
        return False

    def visit_int(self, i: IntType, arg: None) -> bool:
        return False

    def visit_float(self, f: FloatType, arg: None) -> bool:
        return False

    def visit_str(self, s: StrType, arg: None) -> bool:
        return False

    def visit_dtype(self, dtype: DType, arg: None) -> bool:
        return False

    def visit_tuple(self, tup: TupleType, arg: None) -> bool:
        return any(map(lambda f: self.visit(f, None), tup.field_ty_))

    def visit_list(self, lst: ListType, arg: None) -> bool:
        return self.visit(lst.elem_ty_, None)

    def visit_var(self, var: TyVar, arg: None) -> bool:
        return True


class UnificationError(Exception):
    def __init__(self, lhs: Type, rhs: Type):
        self.lhs = lhs
        self.rhs = rhs

    def __str__(self):
        return f'Cannot unify {self.lhs} and {self.rhs}.'


class TypeUnifier(TypeVisitor[Type, Type]):
    """
    Unification for two types, possibly with type variables.
    """

    def visit(self, ty: Type, other: Type) -> Type:
        if ty.kind != TypeKind.var and other.kind == TypeKind.var:
            return self.visit_var(cast(TyVar, other), ty)
        else:
            return super().visit(ty, other)

    def visit_bool(self, b: BoolType, other: Type) -> Type:
        return self._unify_prim(b, other)

    def visit_int(self, i: IntType, other: Type) -> Type:
        return self._unify_prim(i, other)

    def visit_float(self, f: FloatType, other: Type) -> Type:
        return self._unify_prim(f, other)

    def visit_str(self, s: StrType, other: Type) -> Type:
        return self._unify_prim(s, other)

    def visit_dtype(self, dtype: DType, other: Type) -> Type:
        return self._unify_prim(dtype, other)

    @staticmethod
    def _unify_prim(this: Type, other: Type) -> Type:
        if this.kind != other.kind:
            raise UnificationError(this, other)
        return this

    def visit_tuple(self, tup: TupleType, other: Type) -> Type:
        if other.kind == TypeKind.tuple:
            other = cast(TupleType, other)
            if len(tup.field_ty_) != len(other.field_ty_):
                raise UnificationError(tup, other)
            field_ty = map(lambda p: self.visit(p[0], p[1]), zip(tup.field_ty_, other.field_ty_))
            return TupleType(*field_ty)
        elif other.kind == TypeKind.list:
            if not tup.is_homo_:
                raise UnificationError(tup, other)
            return self.visit_list(ListType(unwrap_or(tup.elem_type, TyVar())), other)
        else:
            raise UnificationError(tup, other)

    def visit_list(self, lst: ListType, other: Type) -> Type:
        if other.kind == TypeKind.tuple:
            return self.visit_tuple(cast(TupleType, other), lst)
        elif other.kind != TypeKind.list:
            raise UnificationError(lst, other)
        else:
            other = cast(ListType, other)
            elem_ty = self.visit(lst.elem_ty_, other.elem_ty_)
            return ListType(elem_ty)

    def visit_var(self, var: TyVar, other: Type) -> Type:
        return other
