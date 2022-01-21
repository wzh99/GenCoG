import typing
from enum import Enum, IntEnum, auto
from typing import Any, Callable, Dict, Optional, Union, Type, List, Generic, TypeVar
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
    NOT = auto()
    AND = auto()
    OR = auto()
    FORALL = auto()
    COND = auto()
    ATTR = auto()
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
    FILTER = auto()
    INSET = auto()
    SUBSET = auto()


class Expr:
    """
    Base class for constraint expression.
    """
    kind: ExprKind

    def __init__(self, sub_expr: List['Expr']):
        self.type_: Optional[ty.Type] = None
        self.sub_expr_ = sub_expr
        self.ref_cnt_ = 0
        for s in self.sub_expr_:
            s.ref_cnt_ += 1

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

    def __mod__(self, other: 'ExprLike'):
        return Arith(ArithOp.MOD, self, other)

    def __rmod__(self, other: 'ExprLike'):
        return Arith(ArithOp.MOD, other, self)

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

    def __xor__(self, other: 'ExprLike'):
        return Cmp(CmpOp.NE, self, other)

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
        super().__init__([])
        self.val_ = val
        self.type_ = ty.type_py_value(val)


class Range(Expr):
    """
    Range [begin, end) of a primitive value.
    """
    kind = ExprKind.RANGE

    valid_type_kinds = [TypeKind.int, TypeKind.float]

    def __init__(self, begin: Optional[ExprLike] = None, end: Optional[ExprLike] = None):
        self.begin_ = util.map_optional(to_expr, begin)
        self.end_ = util.map_optional(to_expr, end)
        super().__init__(util.filter_none([self.begin_, self.end_]))

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

    def __init__(self, t: Optional[ty.Type] = None, ran: Optional[Range] = None):
        self.ran_ = ran
        super().__init__(util.filter_none([self.ran_]))
        self.type_ = t


class Symbol(Expr):
    """
    Symbolic value during constraint definition. It is used as a bounded variable in nested
    constraint expression such as `List`.
    """
    kind = ExprKind.SYMBOL

    def __init__(self):
        super().__init__([])


T = TypeVar('T')


class Env(Generic[T]):
    """
    Environment, mapping from symbol to object.
    """

    def __init__(self, prev: Optional['Env'] = None, sym: Optional[Symbol] = None,
                 val: Optional[T] = None):
        self._prev = prev
        self._sym = sym
        if self._sym is not None and val is None:
            raise ValueError(
                'Value cannot be None if symbol is not None.'
            )
        self._val = val

    def __add__(self, pair: typing.Tuple[Symbol, T]):
        return Env(prev=self, sym=pair[0], val=pair[1])

    def __getitem__(self, sym: Symbol) -> Optional[T]:
        env = self
        while env._sym is not None:
            if env._sym is sym:
                return env._val
            else:
                env = env._prev
        return None

    def __contains__(self, sym: Symbol) -> bool:
        return self[sym] is not None


class ArithOp(Enum):
    ADD = '+'
    SUB = '-'
    MUL = '*'
    DIV = '/'
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
        self.op_ = op
        self.lhs_ = to_expr(lhs)
        self.rhs_ = to_expr(rhs)
        super().__init__([self.lhs_, self.rhs_])


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
            (bool, bool): bool.__eq__,
            (int, int): int.__eq__,
            (str, str): str.__eq__,
        },
        CmpOp.NE: {
            (bool, bool): bool.__ne__,
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
        self.op_ = op
        self.lhs_ = to_expr(lhs)
        self.rhs_ = to_expr(rhs)
        super().__init__([self.lhs_, self.rhs_])


class Not(Expr):
    """
    Boolean negation.
    """
    kind = ExprKind.NOT

    def __init__(self, prop: ExprLike):
        self.prop_ = to_expr(prop)
        super().__init__([self.prop_])


class And(Expr):
    """
    Conjunction of two or more boolean expressions. It is suggested that the constructor is
    directly used when there are more than two clauses.
    """
    kind = ExprKind.AND

    def __init__(self, *clauses: ExprLike):
        if len(clauses) <= 1:
            raise ValueError(
                f'Expect at least two clauses, got {len(clauses)}.'
            )
        self.clauses_ = list(to_expr(e) for e in clauses)
        super().__init__(self.clauses_)


class Or(Expr):
    """
    Disjunction of two or more boolean expressions. It is suggested that the constructor is
    directly used when there are more than two clauses.
    """
    kind = ExprKind.OR

    def __init__(self, *clauses: ExprLike):
        if len(clauses) <= 1:
            raise ValueError(
                f'Expect at least two clauses, got {len(clauses)}.'
            )
        self.clauses_ = list(to_expr(e) for e in clauses)
        super().__init__(self.clauses_)


class ForAll(Expr):
    """
    Universal quantifier for propositions defined in an integer range.
    """
    kind = ExprKind.FORALL

    def __init__(self, ran: Range, body_f: Callable[[Symbol], ExprLike]):
        self.ran_ = ran
        self.idx_ = Symbol()
        self.body_ = to_expr(body_f(self.idx_))
        super().__init__([self.ran_, self.idx_, self.body_])


class Cond(Expr):
    """
    Conditional expression.
    """
    kind = ExprKind.COND

    def __init__(self, pred: ExprLike, true_br: ExprLike, fls_br: ExprLike):
        self.pred_ = to_expr(pred)
        self.tr_br_ = to_expr(true_br)
        self.fls_br_ = to_expr(fls_br)
        super().__init__([self.pred_, self.tr_br_, self.fls_br_])


class GetAttr(Expr):
    """
    Get attribute value from operator.
    """
    kind = ExprKind.ATTR

    def __init__(self, name: str):
        super().__init__([])
        self.name_ = name


def a(name: str):
    return GetAttr(name)
