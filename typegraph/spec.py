from typing import List, Optional, Dict
from warnings import warn

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
        self.attrs = attrs
        self.in_num = in_num
        self.in_ranks = in_ranks
        self.in_dtypes = in_dtypes
        self.in_shapes = in_shapes
        self.extra = extra
        self.out_num = out_num
        self.out_ranks = out_ranks
        self.out_dtypes = out_dtypes
        self.out_shapes = out_shapes

    @property
    def attrs(self):
        return self.attrs_

    @attrs.setter
    def attrs(self, *v: List[Attr]):
        self.attrs_ = v[0]

    @property
    def in_num(self):
        return self.in_num_

    @in_num.setter
    def in_num(self, *v: ExprLike):
        self.in_num_ = to_expr(v[0])

    @property
    def in_ranks(self):
        return self.in_ranks_

    @in_ranks.setter
    def in_ranks(self, *v: ExprLike):
        self.in_ranks_ = to_expr(v[0])

    @property
    def in_dtypes(self):
        return self.in_dtypes_

    @in_dtypes.setter
    def in_dtypes(self, *v: ExprLike):
        self.in_dtypes_ = to_expr(v[0])

    @property
    def in_shapes(self):
        return self.in_shapes_

    @in_shapes.setter
    def in_shapes(self, *v: ExprLike):
        self.in_shapes_ = to_expr(v[0])

    @property
    def extra(self):
        return self.extra_

    @extra.setter
    def extra(self, *v: List[ExprLike]):
        self.extra_ = [to_expr(c) for c in v[0]]

    @property
    def out_num(self):
        return self.out_num_

    @out_num.setter
    def out_num(self, *v: ExprLike):
        self.out_num_ = to_expr(v[0])

    @property
    def out_ranks(self):
        return self.out_ranks_

    @out_ranks.setter
    def out_ranks(self, *v: ExprLike):
        self.out_ranks_ = to_expr(v[0])

    @property
    def out_dtypes(self):
        return self.out_dtypes_

    @out_dtypes.setter
    def out_dtypes(self, *v: ExprLike):
        self.out_dtypes_ = to_expr(v[0])

    @property
    def out_shapes(self):
        return self.out_shapes_

    @out_shapes.setter
    def out_shapes(self, *v: ExprLike):
        self.out_shapes_ = to_expr(v[0])


class Op:
    """
    Specification of an operator.
    """

    def __init__(self, name: str, constr: ConstraintSet):
        self.name_ = name
        self.constr_ = constr
        OpRegistry.register(self)


class OpRegistry:
    """
    Registry for all defined operators.
    """

    table: Dict[str, Op] = {}

    @classmethod
    def register(cls, op: Op):
        if op.name_ in cls.table:
            warn(f'Operator {op.name_} has already been registered.')
        else:
            cls.table[op.name_] = op
