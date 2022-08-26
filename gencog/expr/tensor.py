from enum import Enum

from .basic import Expr, ExprKind, ExprLike, to_expr
from .ty import ListType, INT, DTYPE


class TensorKind(Enum):
    input = 'IN'
    output = 'OUT'


class TensorDesc:
    """
    Description of a tensor by its position in an operator.
    """

    def __init__(self, kind: TensorKind, idx: ExprLike):
        self.kind_ = kind
        self.idx_ = to_expr(idx)

    @property
    def shape(self):
        return Shape(self)

    @property
    def rank(self):
        return Rank(self)

    @property
    def dtype(self):
        return GetDType(self)


class TensorList:
    """
    List of input/output tensors of an operator.
    """

    def __init__(self, kind: TensorKind):
        self.kind_ = kind

    def __getitem__(self, idx: ExprLike):
        return TensorDesc(self.kind_, idx)

    @property
    def num(self):
        return Num(self.kind_)


IN = TensorList(TensorKind.input)
OUT = TensorList(TensorKind.output)


class Num(Expr):
    """
    Number of tensors in input/output list.
    """
    kind = ExprKind.NUM

    def __init__(self, t_kind: TensorKind):
        super().__init__([], ty=INT)
        self.t_kind_ = t_kind


class Shape(Expr):
    """
    Get shape of a tensor.
    """
    kind = ExprKind.SHAPE

    def __init__(self, tensor: TensorDesc):
        super().__init__([tensor.idx_], ty=ListType(INT))
        self.tensor_ = tensor

    @property
    def index(self):
        return self.tensor_.idx_

    @property
    def tensor_kind(self):
        return self.tensor_.kind_


class Rank(Expr):
    """
    Get rank of a tensor.
    """
    kind = ExprKind.RANK

    def __init__(self, tensor: TensorDesc):
        super().__init__([tensor.idx_], ty=INT)
        self.tensor_ = tensor

    @property
    def index(self):
        return self.tensor_.idx_

    @property
    def tensor_kind(self):
        return self.tensor_.kind_


class GetDType(Expr):
    """
    Get data type of a tensor.
    """
    kind = ExprKind.DTYPE

    def __init__(self, tensor: TensorDesc):
        super().__init__([tensor.idx_], ty=DTYPE)
        self.tensor_ = tensor

    @property
    def index(self):
        return self.tensor_.idx_

    @property
    def tensor_kind(self):
        return self.tensor_.kind_


class LayoutIndex(Expr):
    """
    Locate index of a dimension in tensor layout.
    """
    kind = ExprKind.LAYOUT_INDEX

    def __init__(self, layout: ExprLike, dim: ExprLike):
        self.layout_ = to_expr(layout)
        self.dim_ = to_expr(dim)
        super().__init__([self.layout_, self.dim_], ty=INT)


class LayoutMap(Expr):
    """
    Specify a tensor shape in source layout, then map it to the target layout.
    """
    kind = ExprKind.LAYOUT_MAP

    def __init__(self, tgt: ExprLike, src: ExprLike, src_shape: ExprLike):
        self.tgt_ = to_expr(tgt)
        self.src_ = to_expr(src)
        self.src_shape_ = to_expr(src_shape)
        super().__init__([self.tgt_, self.src_, self.src_shape_], ty=ListType(INT))
