from typing import List, Dict, Optional
from warnings import warn

from .expr import Type, ExprLike, to_expr


class Attr:
    """
    Specification of an operator attribute.
    """

    def __init__(self, name: str, expr: ExprLike, ty: Optional[Type] = None):
        self.name_ = name
        self.expr_ = to_expr(expr)
        self.ty_ = ty


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
        return self._attrs

    @attrs.setter
    def attrs(self, *v: List[Attr]):
        self._attrs = v[0]

    @property
    def in_num(self):
        return self._in_num

    @in_num.setter
    def in_num(self, *v: ExprLike):
        self._in_num = to_expr(v[0])

    @property
    def in_ranks(self):
        return self._in_ranks

    @in_ranks.setter
    def in_ranks(self, *v: ExprLike):
        self._in_ranks = to_expr(v[0])

    @property
    def in_dtypes(self):
        return self._in_dtypes

    @in_dtypes.setter
    def in_dtypes(self, *v: ExprLike):
        self._in_dtypes = to_expr(v[0])

    @property
    def in_shapes(self):
        return self._in_shapes

    @in_shapes.setter
    def in_shapes(self, *v: ExprLike):
        self._in_shapes = to_expr(v[0])

    @property
    def extra(self):
        return self._extra

    @extra.setter
    def extra(self, *v: List[ExprLike]):
        self._extra = [to_expr(c) for c in v[0]]

    @property
    def out_num(self):
        return self._out_num

    @out_num.setter
    def out_num(self, *v: ExprLike):
        self._out_num = to_expr(v[0])

    @property
    def out_ranks(self):
        return self._out_ranks

    @out_ranks.setter
    def out_ranks(self, *v: ExprLike):
        self._out_ranks = to_expr(v[0])

    @property
    def out_dtypes(self):
        return self._out_dtypes

    @out_dtypes.setter
    def out_dtypes(self, *v: ExprLike):
        self._out_dtypes = to_expr(v[0])

    @property
    def out_shapes(self):
        return self._out_shapes

    @out_shapes.setter
    def out_shapes(self, *v: ExprLike):
        self._out_shapes = to_expr(v[0])


class Op:
    """
    Specification of an operator.
    """

    def __init__(self, name: str, spec: ConstraintSet):
        self.name_ = name
        self.spec_ = spec
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
