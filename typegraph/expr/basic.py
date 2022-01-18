import typing
from enum import Enum, IntEnum, auto
from typing import Any, Callable, Dict, Optional, Union, Type
from warnings import warn

from . import ty
from .ty import TypeKind, ValueType, DataType
from .. import util


class ExprKind(IntEnum):
    # Basic
    CONST = auto()
    VAR = auto()
    SYMBOL = auto()
    RANGE = auto()
    ARITH = auto()
    CMP = auto()
    AND = auto()
    OR = auto()
    FOR_EACH = auto()
    COND = auto()
    GETATTR = auto()
    # Tensor
    NUM = auto()
    SHAPE = auto()
    RANK = auto()
    DTYPE = auto()
    # Array
    TUPLE = auto()
    LIST = auto()
    GETITEM = auto()
    LEN = auto()
    CONCAT = auto()
    SLICE = auto()
    MAP = auto()
    REDUCE_ARRAY = auto()
    REDUCE_INDEX = auto()


class Expr:
    """
    Base class for constraint expression.
    """
    kind: ExprKind

    def __init__(self):
        self.type_: Optional[ty.Type] = None

    def __add__(self, other: 'ExprLike'):
        return Arith(ArithOp.ADD, self, other)

    def __radd__(self, other: 'ExprLike'):
        return Arith(ArithOp.ADD, other, self)

    def __sub__(self, other: 'ExprLike'):
        return Arith(ArithOp.SUB, self, other)

    def __rsub__(self, other: 'ExprLike'):
        return Arith(ArithOp.SUB, other, self)

    def __mul__(self, other: 'ExprLike'):
        return Arith(ArithOp.MUL, self, other)

    def __rmul__(self, other: 'ExprLike'):
        return Arith(ArithOp.MUL, other, self)

    def __truediv__(self, other: 'ExprLike'):
        return Arith(ArithOp.DIV, self, other)

    def __rtruediv__(self, other: 'ExprLike'):
        return Arith(ArithOp.DIV, other, self)

    def __floordiv__(self, other: 'ExprLike'):
        return Arith(ArithOp.DIV, self, other)

    def __rfloordiv__(self, other: 'ExprLike'):
        return Arith(ArithOp.DIV, other, self)

    def max(self, other: 'ExprLike'):
        return Arith(ArithOp.MAX, self, other)

    def min(self, other: 'ExprLike'):
        return Arith(ArithOp.MIN, self, other)

    def __eq__(self, other: 'ExprLike'):
        return Cmp(CmpOp.EQ, self, other)

    def __ne__(self, other: 'ExprLike'):
        return Cmp(CmpOp.NE, self, other)

    def __lt__(self, other: 'ExprLike'):
        return Cmp(CmpOp.LT, self, other)

    def __le__(self, other: 'ExprLike'):
        return Cmp(CmpOp.LE, self, other)

    def __gt__(self, other: 'ExprLike'):
        return Cmp(CmpOp.GT, self, other)

    def __ge__(self, other: 'ExprLike'):
        return Cmp(CmpOp.GE, self, other)

    def __and__(self, other: 'ExprLike'):
        return And(self, other)

    def __or__(self, other: 'ExprLike'):
        return Or(self, other)

    def __getitem__(self, item: 'ExprLike'):
        from .array import GetItem, Slice
        if isinstance(item, Range):
            return Slice(self, item)
        else:
            return GetItem(self, item)


ExprLike = Union[Expr, ValueType]


def to_expr(e: ExprLike) -> Expr:
    """
    Convert a Python object to a constraint expression.

    :param e: The Python object to be converted.
    :return: The converted result.
    """
    from .array import Tuple

    if isinstance(e, Expr):
        return e
    elif isinstance(e, (bool, int, float, str, DataType)):
        return Const(e)
    elif isinstance(e, (tuple, list)):
        return Tuple(*e)
    else:
        raise TypeError(
            'Cannot convert Python object of type {} to constraint expression.'.format(
                util.cls_name(e))
        )


class Const(Expr):
    """
    Constant whose value is known during constraint definition.
    """
    kind = ExprKind.CONST

    def __init__(self, val: ValueType):
        super().__init__()
        self.val_ = val
        self.type_ = ty.type_py_value(val)


class Range(Expr):
    """
    Range [begin, end) of a primitive value.
    """
    kind = ExprKind.RANGE

    valid_type_kinds = [TypeKind.int, TypeKind.float]

    def __init__(self, begin: Optional[ExprLike] = None, end: Optional[ExprLike] = None):
        super().__init__()
        self.begin_ = util.map_optional(to_expr, begin)
        self.end_ = util.map_optional(to_expr, end)

    @classmethod
    def validate_type(cls, t: ty.Type, ran: Optional['Range']) -> Optional['Range']:
        if t.kind in Range.valid_type_kinds:
            return ran
        elif ran is not None:
            warn(f'Ignore range for type {t}.')
        return None


class Var(Expr):
    """
    Variable whose value is unknown and should be figured out during constraint solving. A new
    constraint variable will be created each time a `Var` is evaluated during constraint
    instantiation.
    """
    kind = ExprKind.VAR

    def __init__(self, t: ty.Type, ran: Optional[Range] = None):
        super().__init__()
        if t.kind not in ty.Type.prim_kinds:
            raise ValueError(
                'Cannot create variable for non-primitive type {}'.format(
                    util.cls_name(t))
            )
        self.type_ = t
        self.range_ = Range.validate_type(t, ran)


class Symbol(Expr):
    """
    Symbolic value during constraint definition. It is used as a bounded variable in nested
    constraint expression such as `List`.
    """
    kind = ExprKind.SYMBOL


class ArithOp(Enum):
    ADD = '+'
    SUB = '-'
    MUL = '*'
    DIV = '/'  # __floordiv__ for int, __div__ for float
    MOD = '%'
    MAX = 'max'
    MIN = 'min'


class Arith(Expr):
    """
    Arithmetic expressions.
    """
    kind = ExprKind.ARITH

    op_funcs: Dict[ArithOp, Dict[typing.Tuple[Type, Type], Callable[[Any, Any], Any]]] = {
        ArithOp.ADD: {
            (int, int): int.__add__,
        },
        ArithOp.SUB: {
            (int, int): int.__sub__,
        },
        ArithOp.MUL: {
            (int, int): int.__mul__,
        },
        ArithOp.DIV: {
            (int, int): int.__floordiv__,
        },
        ArithOp.MOD: {
            (int, int): int.__mod__,
        },
        ArithOp.MAX: {
            (int, int): max,
        },
        ArithOp.MIN: {
            (int, int): min,
        },
    }

    def __init__(self, op: ArithOp, lhs: ExprLike, rhs: ExprLike):
        super().__init__()
        self.op_ = op
        self.lhs_ = to_expr(lhs)
        self.rhs_ = to_expr(rhs)


class CmpOp(Enum):
    EQ = '='
    NE = '!='
    LT = '<'
    LE = '<='
    GT = '>'
    GE = '>='


class Cmp(Expr):
    """
    Comparison expressions.
    """
    kind = ExprKind.CMP

    op_funcs: Dict[CmpOp, Dict[typing.Tuple[Type, Type], Callable[[Any, Any], bool]]] = {
        CmpOp.EQ: {
            (int, int): int.__eq__,
            (str, str): str.__eq__,
        },
        CmpOp.NE: {
            (int, int): int.__ne__,
            (str, str): str.__ne__,
        },
        CmpOp.LT: {
            (int, int): int.__lt__,
        },
        CmpOp.LE: {
            (int, int): int.__le__,
        },
        CmpOp.GT: {
            (int, int): int.__gt__,
        },
        CmpOp.GE: {
            (int, int): int.__ge__,
        },
    }

    def __init__(self, op: CmpOp, lhs: ExprLike, rhs: ExprLike):
        super().__init__()
        self.op_ = op
        self.lhs_ = to_expr(lhs)
        self.rhs_ = to_expr(rhs)


class And(Expr):
    """
    Conjunction of two or more boolean expressions. It is suggested that the constructor is
    directly used when there are more than two clauses.
    """
    kind = ExprKind.AND

    def __init__(self, *clauses: ExprLike):
        super().__init__()
        if len(clauses) <= 1:
            raise ValueError(
                f'Expect at least two clauses, got {len(clauses)}.'
            )
        self.clauses_ = tuple(to_expr(e) for e in clauses)


class Or(Expr):
    """
    Disjunction of two or more boolean expressions. It is suggested that the constructor is
    directly used when there are more than two clauses.
    """
    kind = ExprKind.OR

    def __init__(self, *clauses: ExprLike):
        super().__init__()
        if len(clauses) <= 1:
            raise ValueError(
                f'Expect at least two clauses, got {len(clauses)}.'
            )
        self.clauses_ = tuple(to_expr(e) for e in clauses)


class ForEach(Expr):
    """
    Universal quantifier for propositions defined in an integer range.
    """
    kind = ExprKind.FOR_EACH

    def __init__(self, ran: Range, body_f: Callable[[Symbol], ExprLike]):
        super().__init__()
        self.ran_ = ran
        self.idx_ = Symbol()
        self.body_ = to_expr(body_f(self.idx_))


class Cond(Expr):
    """
    Conditional expression.
    """
    kind = ExprKind.COND

    def __init__(self, pred: ExprLike, true_br: ExprLike, fls_br: ExprLike):
        super().__init__()
        self.pred_ = to_expr(pred)
        self.true_br_ = to_expr(true_br)
        self.fls_br_ = to_expr(fls_br)


class GetAttr(Expr):
    """
    Get attribute value from operator.
    """
    kind = ExprKind.GETATTR

    def __init__(self, name: str):
        super().__init__()
        self.name_ = name
