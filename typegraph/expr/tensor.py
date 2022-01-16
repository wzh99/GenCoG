from enum import IntEnum, auto
from typing import Dict, Callable
from .base import Expr, ExprKind, ExprLike, to_expr


class TensorKind(IntEnum):
    INPUT = auto()
    OUTPUT = auto()


class TensorSpec:
    """
    Specifies a tensor by its position in an operator.
    """

    def __init__(self, kind: TensorKind, idx: ExprLike):
        self.kind_ = kind
        self.idx_ = to_expr(idx)

    avail_attrs: Dict[str, Callable[['TensorSpec'], Expr]] = {
        'shape': lambda self: Shape(self),
        'rank': lambda self: Rank(self),
        'dtype': lambda self: DType(self),
    }

    def __getattr__(self, name: str):
        if name in self.avail_attrs:
            return self.avail_attrs[name](self)
        else:
            raise AttributeError(
                'Unknown attribute \'{}\''.format(name)
            )


class TensorList:
    def __init__(self, kind: TensorKind):
        self.kind_ = kind

    def __getitem__(self, idx: ExprLike):
        return TensorSpec(self.kind_, idx)


i = TensorList(TensorKind.INPUT)
o = TensorList(TensorKind.OUTPUT)


class Shape(Expr):
    """
    Get shape of a tensor.
    """
    kind = ExprKind.SHAPE

    def __init__(self, tensor: TensorSpec):
        super().__init__()
        self.tensor_ = tensor


class Rank(Expr):
    """
    Get rank of a tensor.
    """
    kind = ExprKind.RANK

    def __init__(self, tensor: TensorSpec):
        super().__init__()
        self.tensor_ = tensor


class DType(Expr):
    """
    Get data type of a tensor.
    """
    kind = ExprKind.DTYPE

    def __init__(self, tensor: TensorSpec):
        super().__init__()
        self.tensor_ = tensor
