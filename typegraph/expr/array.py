from typing import Callable

from .basic import Expr, ExprKind, ExprLike, Symbol, Range, ArithOp, to_expr


class Tuple(Expr):
    """
    Create a fixed-length array of possibly heterogeneous elements.
    """
    kind = ExprKind.TUPLE

    def __init__(self, *fields: ExprLike):
        super().__init__()
        if len(fields) == 0:
            raise ValueError(
                f'Expect at least one field, got {len(fields)}.'
            )
        self.fields_ = tuple(to_expr(f) for f in fields)


class List(Expr):
    """
    Create a variable-length array of homogeneous elements.
    """
    kind = ExprKind.LIST

    def __init__(self, length: ExprLike, body_f: Callable[[Symbol], ExprLike]):
        super().__init__()
        self.len_ = to_expr(length)
        self.idx_ = Symbol()
        self.body_ = to_expr(body_f(self.idx_))


class GetItem(Expr):
    """
    Get one item from an array.
    """
    kind = ExprKind.GETITEM

    def __init__(self, arr: ExprLike, idx: ExprLike):
        super().__init__()
        self.arr_ = to_expr(arr)
        self.idx_ = to_expr(idx)


class Len(Expr):
    """
    Get length of an array.
    """
    kind = ExprKind.LEN

    def __init__(self, arr: ExprLike):
        super().__init__()
        self.arr_ = to_expr(arr)


class Concat(Expr):
    """
    Concatenate two or more arrays.
    """
    kind = ExprKind.CONCAT

    def __init__(self, *arrays: ExprLike):
        super().__init__()
        if len(arrays) <= 1:
            raise ValueError(
                f'Expect at least two arrays, got {len(arrays)}.'
            )
        self.arrays_ = tuple(to_expr(a) for a in arrays)


class Slice(Expr):
    """
    Get slice from an array.
    """

    def __init__(self, arr: ExprLike, ran: Range):
        super().__init__()
        self.arr_ = to_expr(arr)
        self.ran_ = ran


class Map(Expr):
    """
    Map elements in a list by a given function.
    """
    kind = ExprKind.MAP

    def __init__(self, arr: ExprLike, f: Callable[[Symbol], ExprLike]):
        super().__init__()
        self.arr_ = to_expr(arr)
        self.sym_ = Symbol()
        self.body_ = to_expr(f(self.sym_))


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

    def __init__(self, arr: ExprLike, op: ArithOp, init: ExprLike):
        super().__init__()
        self.arr_ = to_expr(arr)
        if op not in REDUCE_OPS:
            raise ValueError(
                f'Operator {op} cannot be used for reduction.'
            )
        self.op_ = op
        self.init_ = to_expr(init)


class ReduceIndex(Expr):
    """
    Reduce expressions in an integer range.
    """
    kind = ExprKind.REDUCE_INDEX

    def __init__(self, ran: Range, op: ArithOp, body_f: Callable[[Symbol], ExprLike],
                 init: ExprLike):
        super().__init__()
        self.ran_ = ran
        if op not in REDUCE_OPS:
            raise ValueError(
                f'Operator {op} cannot be used for reduction.'
            )
        self.op_ = op
        self.idx_ = Symbol()
        self.body_ = to_expr(body_f(self.idx_))
        self.init_ = to_expr(init)


class Filter(Expr):
    """
    Filter elements according to a predicate.
    """
    kind = ExprKind.FILTER

    def __init__(self, arr: ExprLike, pred_f: Callable[[Symbol], ExprLike]):
        super().__init__()
        self.arr_ = to_expr(arr)
        self.sym_ = Symbol()
        self.pred_ = to_expr(pred_f(self.sym_))


class InSet(Expr):
    """
    Whether a value is in a set represented by an array.
    """
    kind = ExprKind.INSET

    def __init__(self, elem: ExprLike, s: ExprLike):
        super().__init__()
        self.elem_ = to_expr(elem)
        self.set_ = to_expr(s)


class Subset(Expr):
    """
    Whether a set is subset of another set.
    """
    kind = ExprKind.SUBSET

    def __init__(self, sub: ExprLike, sup: ExprLike):
        super().__init__()
        self.sub_ = to_expr(sub)
        self.sup_ = to_expr(sup)
