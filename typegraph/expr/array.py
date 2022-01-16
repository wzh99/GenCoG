from .base import Expr, ExprKind, ExprLike, Symbol, Range, to_expr
from typing import Callable


class Tuple(Expr):
    """
    Create a fixed-length array of possibly heterogeneous elements.
    """
    kind = ExprKind.TUPLE

    def __init__(self, *fields: ExprLike):
        super().__init__()
        if len(fields) == 0:
            raise ValueError(
                'At least one field must be provided.'
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
                'At least two arrays must be provided.'
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
