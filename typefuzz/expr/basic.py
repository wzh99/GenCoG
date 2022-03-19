import typing as t
from enum import Enum, IntEnum, auto
from typing import Any, Callable, Dict, Optional, Union, List, Iterator, Iterable, TypeVar
from warnings import warn

from .ty import Type, TypeKind, ValueType, DataType, type_py_value, BOOL, INT, STR
from ..util import cls_name, map_opt, filter_none, unwrap, unwrap_or


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
    DUMMY = auto()
    # Tensor
    NUM = auto()
    SHAPE = auto()
    RANK = auto()
    DTYPE = auto()
    LAYOUT_INDEX = auto()
    LAYOUT_MAP = auto()
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
    PERM = auto()


class Expr:
    """
    Base class for constraint expression.
    """
    kind: ExprKind

    def __init__(self, sub_expr: List['Expr'], ty: Optional[Type] = None):
        self.type_: Optional[Type] = ty
        self.sub_expr_ = sub_expr

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
        elif isinstance(item, slice):
            from . import Len
            return Slice(self, Range(begin=unwrap_or(item.start, 0),
                                     end=unwrap_or(item.stop, Len(self))))
        else:
            return GetItem(self, item)

    def __repr__(self):
        from ..expr.fmt import print_expr
        from ..util import CodeBuffer

        buf = CodeBuffer()
        print_expr(self, buf, [])
        return str(buf)


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
            f'Cannot convert Python object of type {cls_name(e)} to constraint expression.'
        )


class Const(Expr):
    """
    Constant whose value is known during constraint definition.
    """
    kind = ExprKind.CONST

    def __init__(self, val: ValueType):
        super().__init__([])
        self.val_ = val
        self.type_ = type_py_value(val)
        if not self.type_.is_scalar:
            raise TypeError('Cannot create constant of non-scalar type.')


class Range(Expr):
    """
    Range [begin, end) of a primitive value.
    """
    kind = ExprKind.RANGE

    valid_type_kinds = [TypeKind.int, TypeKind.float]

    def __init__(self, begin: Optional[ExprLike] = 0, end: Optional[ExprLike] = None,
                 ty: Optional[Type] = None):
        self.begin_ = map_opt(to_expr, begin)
        self.end_ = map_opt(to_expr, end)
        super().__init__(filter_none([self.begin_, self.end_]), ty=ty)

    def require_begin(self):
        if self.begin_ is None:
            raise ValueError('Range begin cannot be None.')

    def require_end(self):
        if self.end_ is None:
            raise ValueError('Range end cannot be None.')

    def require_both(self):
        self.require_begin()
        self.require_end()


def iran(begin: ExprLike, end: ExprLike):
    return Range(begin, end + 1)


class Var(Expr):
    """
    Variable whose value is unknown and should be figured out during constraint solving. A new
    constraint variable will be created each time a `Var` is evaluated during constraint
    instantiation.
    """
    kind = ExprKind.VAR

    def __init__(self, ty: Optional[Type] = None, ran: Optional[Range] = None,
                 choices: Optional[ExprLike] = None, tmpl: bool = False):
        self.ran_ = ran
        self.choices_ = map_opt(to_expr, choices)
        if self.choices_ is not None and self.ran_ is not None:
            warn('Choices and range are both provided. Range is ignored.')
            self.ran_ = None
        self.tmpl_ = tmpl
        # noinspection PyTypeChecker
        super().__init__(filter_none([self.ran_, self.choices_]))
        if ty is not None:
            self.type_ = ty


class Symbol(Expr):
    """
    Symbolic value during constraint definition. It is used as a bounded variable in nested
    constraint expression such as `List`.
    """
    kind = ExprKind.SYMBOL

    def __init__(self, ty: Optional[Type] = None):
        super().__init__([], ty=ty)


T = TypeVar('T')


class Env(Iterable[T]):
    """
    Environment, mapping from symbol to object.
    """

    def __init__(self, prev: Optional['Env'] = None, sym: Optional[Symbol] = None,
                 val: Optional[T] = None):
        self.prev_ = prev
        self.sym_ = sym
        if self.prev_ is not None and sym is None:
            raise ValueError(
                'Cannot create empty mapping.'
            )
        if self.sym_ is not None and val is None:
            raise ValueError(
                'Value cannot be None if symbol is not None.'
            )
        self.val_ = val

    @property
    def empty(self):
        return self.prev_ is None

    def __add__(self, pair: t.Tuple[Symbol, T]):
        return Env(prev=self, sym=pair[0], val=pair[1])

    def __getitem__(self, sym: Symbol) -> Optional[T]:
        env = self
        while env.sym_ is not None:
            if env.sym_ is sym:
                return env.val_
            else:
                env = env.prev_
        return None

    def __contains__(self, sym: Symbol) -> bool:
        return self[sym] is not None

    def __iter__(self):
        return EnvIter(self)


class EnvIter(Iterator[T]):
    def __init__(self, env: Env[T]):
        self.ref = env

    def __next__(self):
        if self.ref.empty:
            raise StopIteration()
        else:
            pair = unwrap(self.ref.sym_), unwrap(self.ref.val_)
            self.ref = self.ref.prev_
            return pair

    def __iter__(self):
        return self


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

    op_funcs: Dict[ArithOp, Dict[Type, Callable[[Any, Any], Any]]] = {
        ArithOp.ADD: {
            INT: int.__add__,
        },
        ArithOp.SUB: {
            INT: int.__sub__,
        },
        ArithOp.MUL: {
            INT: int.__mul__,
        },
        ArithOp.DIV: {
            INT: int.__floordiv__,
        },
        ArithOp.MOD: {
            INT: int.__mod__,
        },
        ArithOp.MAX: {
            INT: max,
        },
        ArithOp.MIN: {
            INT: min,
        },
    }

    def __init__(self, op: ArithOp, lhs: ExprLike, rhs: ExprLike, ty: Optional[Type] = None):
        self.op_ = op
        self.lhs_ = to_expr(lhs)
        self.rhs_ = to_expr(rhs)
        super().__init__([self.lhs_, self.rhs_], ty=ty)


class CmpOp(Enum):
    EQ = '=='
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

    op_funcs: Dict[CmpOp, Dict[Type, Callable[[Any, Any], bool]]] = {
        CmpOp.EQ: {
            BOOL: bool.__eq__,
            INT: int.__eq__,
            STR: str.__eq__,
        },
        CmpOp.NE: {
            BOOL: bool.__ne__,
            INT: int.__ne__,
            STR: str.__ne__,
        },
        CmpOp.LT: {
            INT: int.__lt__,
        },
        CmpOp.LE: {
            INT: int.__le__,
        },
        CmpOp.GT: {
            INT: int.__gt__,
        },
        CmpOp.GE: {
            INT: int.__ge__,
        },
    }

    def __init__(self, op: CmpOp, lhs: ExprLike, rhs: ExprLike):
        self.op_ = op
        self.lhs_ = to_expr(lhs)
        self.rhs_ = to_expr(rhs)
        super().__init__([self.lhs_, self.rhs_], ty=BOOL)


class Not(Expr):
    """
    Boolean negation.
    """
    kind = ExprKind.NOT

    def __init__(self, prop: ExprLike):
        self.prop_ = to_expr(prop)
        super().__init__([self.prop_], ty=BOOL)


class And(Expr):
    """
    Conjunction of two or more boolean expressions. It is suggested that the constructor is
    directly used when there are more than two clauses.
    """
    kind = ExprKind.AND

    def __init__(self, *clauses: ExprLike):
        self.clauses_ = list(to_expr(e) for e in clauses)
        super().__init__(self.clauses_, ty=BOOL)


class Or(Expr):
    """
    Disjunction of two or more boolean expressions. It is suggested that the constructor is
    directly used when there are more than two clauses.
    """
    kind = ExprKind.OR

    def __init__(self, *clauses: ExprLike):
        self.clauses_ = list(to_expr(e) for e in clauses)
        super().__init__(self.clauses_, ty=BOOL)


class ForAll(Expr):
    """
    Universal quantifier for propositions defined in an integer range.
    """
    kind = ExprKind.FORALL

    def __init__(self, ran: Range, body_f: Optional[Callable[[Symbol], ExprLike]] = None,
                 idx: Optional[Symbol] = None, body: Optional[Expr] = None):
        ran.require_both()
        self.ran_ = ran
        if body_f is not None:
            self.idx_ = Symbol()
            self.body_ = to_expr(body_f(self.idx_))
        else:
            assert idx is not None and body is not None
            self.idx_ = idx
            self.body_ = body
        super().__init__([self.ran_, self.idx_, self.body_], ty=BOOL)


class Cond(Expr):
    """
    Conditional expression.
    """
    kind = ExprKind.COND

    def __init__(self, pred: ExprLike, true_br: ExprLike, fls_br: ExprLike,
                 ty: Optional[Type] = None):
        self.pred_ = to_expr(pred)
        self.tr_br_ = to_expr(true_br)
        self.fls_br_ = to_expr(fls_br)
        super().__init__([self.pred_, self.tr_br_, self.fls_br_], ty=ty)


class GetAttr(Expr):
    """
    Get attribute value from operator.
    """
    kind = ExprKind.ATTR

    def __init__(self, name: str, ty: Optional[Type] = None):
        super().__init__([], ty=ty)
        self.name_ = name


def a(name: str):
    return GetAttr(name)


class Dummy(Expr):
    """
    Indicate unknown expression.
    """
    kind = ExprKind.DUMMY

    def __init__(self):
        super().__init__([])
