from typing import Callable, Optional

from .basic import Expr, ExprKind, ExprLike, Symbol, Range, ArithOp, to_expr
from .ty import Type, INT, BOOL


class Tuple(Expr):
    """
    Create a fixed-length array of possibly heterogeneous elements.
    """
    kind = ExprKind.TUPLE

    def __init__(self, *fields: ExprLike, ty: Optional[Type] = None):
        self.fields_ = [to_expr(f) for f in fields]
        super().__init__(self.fields_, ty=ty)


class List(Expr):
    """
    Create a variable-length array of homogeneous elements.
    """
    kind = ExprKind.LIST

    def __init__(self, length: ExprLike, body_f: Optional[Callable[[Symbol], ExprLike]] = None,
                 idx: Optional[Symbol] = None, body: Optional[Expr] = None,
                 ty: Optional[Type] = None):
        self.len_ = to_expr(length)
        if body_f is not None:
            self.idx_ = Symbol()
            self.body_ = to_expr(body_f(self.idx_))
        else:
            assert idx is not None and body is not None
            self.idx_ = idx
            self.body_ = body
        super().__init__([self.len_, self.idx_, self.body_], ty=ty)


class GetItem(Expr):
    """
    Get one item from an array.
    """
    kind = ExprKind.GETITEM

    def __init__(self, arr: ExprLike, idx: ExprLike, ty: Optional[Type] = None):
        self.arr_ = to_expr(arr)
        self.idx_ = to_expr(idx)
        super().__init__([self.arr_, self.idx_], ty=ty)


class Len(Expr):
    """
    Get length of an array.
    """
    kind = ExprKind.LEN

    def __init__(self, arr: ExprLike):
        self.arr_ = to_expr(arr)
        super().__init__([self.arr_], ty=INT)


class Concat(Expr):
    """
    Concatenate two or more arrays.
    """
    kind = ExprKind.CONCAT

    def __init__(self, *arrays: ExprLike, ty: Optional[Type] = None):
        if len(arrays) <= 1:
            raise ValueError(
                f'Expect at least two arrays, got {len(arrays)}.'
            )
        self.arrays_ = list(to_expr(a) for a in arrays)
        super().__init__(self.arrays_, ty=ty)


class Slice(Expr):
    """
    Get slice from an array.
    """
    kind = ExprKind.SLICE

    def __init__(self, arr: ExprLike, ran: Range, ty: Optional[Type] = None):
        ran.require_both()
        self.arr_ = to_expr(arr)
        self.ran_ = ran
        super().__init__([self.arr_, self.ran_], ty=ty)


class Map(Expr):
    """
    Map elements in a list by a given function.
    """
    kind = ExprKind.MAP

    def __init__(self, arr: ExprLike, body_f: Optional[Callable[[Symbol], ExprLike]] = None,
                 sym: Optional[Symbol] = None, body: Optional[Expr] = None,
                 ty: Optional[Type] = None):
        self.arr_ = to_expr(arr)
        if body_f is not None:
            self.sym_ = Symbol()
            self.body_ = to_expr(body_f(self.sym_))
        else:
            assert sym is not None and body is not None
            self.sym_ = sym
            self.body_ = body
        super().__init__([self.arr_, self.sym_, self.body_], ty=ty)


REDUCE_OPS = [
    ArithOp.ADD,
    ArithOp.MUL,
    ArithOp.MAX,
    ArithOp.MIN,
]


class ReduceArray(Expr):
    """
    Reduce elements in an array.
    """
    kind = ExprKind.REDUCE_ARRAY

    def __init__(self, arr: ExprLike, op: ArithOp, init: ExprLike, ty: Optional[Type] = None):
        self.arr_ = to_expr(arr)
        if op not in REDUCE_OPS:
            raise ValueError(
                f'Operator {op} cannot be used for reduction.'
            )
        self.op_ = op
        self.init_ = to_expr(init)
        super().__init__([self.arr_, self.init_], ty=ty)


class ReduceRange(Expr):
    """
    Reduce expressions in an integer range.
    """
    kind = ExprKind.REDUCE_INDEX

    def __init__(self, ran: Range, op: ArithOp, init: ExprLike,
                 body_f: Optional[Callable[[Symbol], ExprLike]] = None,
                 idx: Optional[Symbol] = None, body: Optional[Expr] = None,
                 ty: Optional[Type] = None):
        ran.require_both()
        self.ran_ = ran
        if op not in REDUCE_OPS:
            raise ValueError(
                f'Operator {op} cannot be used for reduction.'
            )
        self.op_ = op
        self.init_ = to_expr(init)
        if body_f is not None:
            self.idx_ = Symbol()
            self.body_ = to_expr(body_f(self.idx_))
        else:
            assert idx is not None and body is not None
            self.idx_ = idx
            self.body_ = body
        super().__init__([self.idx_, self.body_, self.init_], ty=ty)


class Filter(Expr):
    """
    Filter elements according to a predicate.
    """
    kind = ExprKind.FILTER

    def __init__(self, arr: ExprLike, pred_f: Optional[Callable[[Symbol], ExprLike]] = None,
                 sym: Optional[Symbol] = None, pred: Optional[Expr] = None,
                 ty: Optional[Type] = None):
        self.arr_ = to_expr(arr)
        if pred_f is not None:
            self.sym_ = Symbol()
            self.pred_ = to_expr(pred_f(self.sym_))
        else:
            assert sym is not None and pred is not None
            self.sym_ = sym
            self.pred_ = pred
        super().__init__([self.arr_, self.sym_, self.pred_], ty=ty)


class InSet(Expr):
    """
    Whether a value is in a set represented by an array.
    """
    kind = ExprKind.INSET

    def __init__(self, elem: ExprLike, s: ExprLike):
        self.elem_ = to_expr(elem)
        self.set_ = to_expr(s)
        super().__init__([self.elem_, self.set_], ty=BOOL)


class Subset(Expr):
    """
    Whether a set is subset of another set.
    """
    kind = ExprKind.SUBSET

    def __init__(self, sub: ExprLike, sup: ExprLike):
        self.sub_ = to_expr(sub)
        self.sup_ = to_expr(sup)
        super().__init__([self.sub_, self.sup_], ty=BOOL)


class Perm(Expr):
    """
    Whether an array is a permutation of another.
    """
    kind = ExprKind.PERM

    def __init__(self, tgt: ExprLike, src: ExprLike):
        self.tgt_ = to_expr(tgt)
        self.src_ = to_expr(src)
        super().__init__([self.tgt_, self.src_], ty=BOOL)
