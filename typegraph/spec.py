from typing import List, Optional

from .expr import Type, ExprLike, Range, to_expr


class Attr:
    """
    Specification of an operator attribute.
    """

    def __init__(self, name: str, ty: Type, ran: Optional[Range] = None):
        self.name_ = name
        self.ty_ = ty
        self.range_ = Range.validate_type(ty, ran)


class ConstraintSet:
    """
    Set of constraints specified for an operator. This object can be shared across different
    operators if they have identical attributes and typing constraints.
    """

    def __init__(self,
                 attrs: List[Attr],
                 in_num: ExprLike,
                 in_ranks: ExprLike,
                 in_dtypes: ExprLike,
                 in_shapes: ExprLike,
                 extra: List[ExprLike],
                 out_num: ExprLike,
                 out_ranks: ExprLike,
                 out_dtypes: ExprLike,
                 out_shapes: ExprLike):
        """
        :param attrs:
        """
        self.attrs_ = attrs
        self.in_num_ = to_expr(in_num)
        self.in_ranks_ = to_expr(in_ranks)
        self.in_dtypes_ = to_expr(in_dtypes)
        self.in_types_ = to_expr(in_shapes)
        self.extra_ = [to_expr(c) for c in extra]
        self.out_num_ = to_expr(out_num)
        self.out_ranks_ = to_expr(out_ranks)
        self.out_dtypes_ = to_expr(out_dtypes)
        self.out_shapes_ = to_expr(out_shapes)


class Op:
    """
    Specification of an operator.
    """

    def __init__(self, name: str, constr: ConstraintSet):
        self.name_ = name
        self.constr_ = constr
