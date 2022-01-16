from enum import IntEnum, auto
from typing import Dict
from typing import Optional, Union

from .. import ty, util


class ExprKind(IntEnum):
    CONST = auto()
    VAR = auto()
    SYMBOL = auto()
    RANGE = auto()
    ARITH = auto()
    CMP = auto()
    AND = auto()
    OR = auto()
    COND = auto()
    TUPLE = auto()
    LIST = auto()
    GETITEM = auto()
    LEN = auto()
    CONCAT = auto()
    SLICE = auto()
    REDUCE_ELEM = auto()
    REDUCE_INDEXED = auto()


class Expr:
    """
    Base class for constraint expression.
    """
    kind: ExprKind

    def __init__(self):
        self.type_: Optional[ty.Type] = None


ExprLike = Union[Expr, ty.PyTypeable]


def cvt_expr(e: ExprLike) -> Expr:
    """
    Convert a Python object to a constraint expression.

    :param e: The Python object to be converted.
    :return: The converted result.
    """
    if isinstance(e, Expr):
        return e
    elif isinstance(e, (int, float, str)):
        return Const(e)
    else:
        raise TypeError(
            'Cannot convert Python object of type {}'.format(
                util.cls_name(e))
        )


class Const(Expr):
    """
    Constant whose value is known during constraint definition.
    """
    kind = ExprKind.CONST

    def __init__(self, val: ty.PyTypeable):
        super().__init__()
        self.val_ = val
        self.type_ = ty.type_py_value(val)


class Range(Expr):
    """
    Range [begin, end) of a primitive value.
    """
    kind = ExprKind.RANGE

    def __init__(self, begin: Optional[ExprLike], end: Optional[ExprLike]):
        super().__init__()
        self.begin_ = util.map_optional(cvt_expr, begin)
        self.end_ = util.map_optional(cvt_expr, end)


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
                'Cannot create variable for non-primitive type \'{}\''.format(
                    util.cls_name(t))
            )
        self.type_ = t
        self.range_ = ran


class Symbol(Expr):
    """
    Symbolic value during constraint definition. It is used as a bounded variable in nested
    constraint expression such as `List`.
    """
    kind = ExprKind.SYMBOL

    _registry: Dict[str, 'Symbol'] = {}

    @classmethod
    def create(cls, name: str):
        """
        Create a new symbol or find previously created symbol according to the given name. Two
        distinct symbols have overlapping scope, they must have different names. Otherwise,
        we do not care whether they share the same name or not.

        :param name: The name used to identify the symbol.
        :return: The result symbol.
        """
        if name not in cls._registry:
            cls._registry[name] = Symbol()
        return cls._registry[name]

    @classmethod
    def clear(cls):
        cls._registry.clear()


def s(name: str):
    """
    Shorthand for `Symbol.create`.
    """
    return Symbol.create(name)
