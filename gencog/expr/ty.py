import typing as t
from enum import IntEnum, auto
from typing import Union, Optional, Dict, Callable, cast

from .. import util


class TypeKind(IntEnum):
    """
    Simple RTTI mechanism for `Type`.
    """
    bool = auto()
    int = auto()
    float = auto()
    str = auto()
    dtype = auto()
    tuple = auto()
    list = auto()
    var = auto()


class Type:
    """
    Base class for all expression types.
    """
    kind: TypeKind

    scalar_kinds = [
        TypeKind.bool,
        TypeKind.int,
        TypeKind.float,
        TypeKind.str,
        TypeKind.dtype,
    ]

    @property
    def is_scalar(self):
        return self.kind in self.scalar_kinds

    @property
    def elem_type(self) -> Optional['Type']:
        return None

    def __eq__(self, other: 'Type'):
        """
        Compare structural equality of two types.
        """
        return self.kind == other.kind

    def __hash__(self):
        return self.kind.__hash__()

    def __repr__(self):
        return self.kind.name


class BoolType(Type):
    """
    Boolean type.
    """
    kind = TypeKind.bool


class IntType(Type):
    """
    Integer type.
    """
    kind = TypeKind.int


class FloatType(Type):
    """
    Float type.
    """
    kind = TypeKind.float


class StrType(Type):
    """
    String type.
    """
    kind = TypeKind.str


class DType(Type):
    """
    Type for tensor data type.
    """
    kind = TypeKind.dtype


BOOL = BoolType()
INT = IntType()
FLOAT = FloatType()
STR = StrType()
DTYPE = DType()


class TypeCode(IntEnum):
    int = auto()
    uint = auto()
    float = auto()
    bfloat = auto()


class DataType:
    def __init__(self, code: TypeCode, bits: int):
        self.code_ = code
        self.bits_ = bits

    @classmethod
    def from_str(cls, s: str):
        for code in TypeCode:
            if s.find(code.name) != 0:
                continue
            bits = int(s[len(code.name):])
            return DataType(TypeCode(code.value), bits)
        raise ValueError(
            f'Cannot create DataType from \'{s}\''
        )

    @classmethod
    def i(cls, bits: int):
        return DataType(TypeCode.int, bits)

    @classmethod
    def u(cls, bits: int):
        return DataType(TypeCode.uint, bits)

    @classmethod
    def b(cls):
        return DataType(TypeCode.uint, 1)

    @classmethod
    def f(cls, bits: int):
        return DataType(TypeCode.float, bits)

    @classmethod
    def bf(cls, bits: int):
        return DataType(TypeCode.bfloat, bits)

    def __eq__(self, other: 'DataType'):
        return self.code_ == other.code_ and self.bits_ == other.bits_

    def __hash__(self):
        return hash(self.code_) ^ hash(self.bits_)

    def __str__(self):
        return self.code_.name + str(self.bits_)


common_dtypes = [
    # DataType.b(),
    DataType.i(8), DataType.i(16), DataType.i(32), DataType.i(64),
    DataType.u(8), DataType.u(16), DataType.u(32), DataType.u(64),
    DataType.f(16), DataType.f(32), DataType.f(64),
    # DataType.bf(16),  # bfloat16 compilation failed in Relay backend
]

float_dtypes = [
    DataType.f(16), DataType.f(32), DataType.f(64),
]


class TupleType(Type):
    """
    Type for fixed-length array of possibly heterogeneous elements. A homogeneous tuple type can
    be cast to a list type.
    """
    kind = TypeKind.tuple

    def __init__(self, *field_ty: Type):
        self.field_ty_ = field_ty
        self.is_homo_ = self._is_homo()

    def __eq__(self, other: Type):
        if other.kind == TypeKind.list:
            other = cast(ListType, other)
            return self.is_homo_ and (
                    len(self.field_ty_) == 0 or self.field_ty_[0] == other.elem_ty_)
        elif self.kind != other.kind:
            return False
        other = cast(TupleType, other)
        if len(self.field_ty_) != len(other.field_ty_):
            return False
        return all(map(lambda p: p[0] == p[1], zip(self.field_ty_, other.field_ty_)))

    def _is_homo(self):
        if len(self.field_ty_) <= 1:
            return True
        return all(map(lambda ty: ty == self.field_ty_[0], self.field_ty_[1:]))

    @property
    def elem_type(self) -> Optional['Type']:
        if (not self.is_homo_) or (len(self.field_ty_) == 0):
            return None
        else:
            return self.field_ty_[0]

    def __str__(self):
        if len(self.field_ty_) == 0:
            return '()'
        if len(self.field_ty_) == 1:
            return '({},)'.format(str(self.field_ty_[0]))
        else:
            return '({})'.format(', '.join(map(lambda f: str(f), self.field_ty_)))


class ListType(Type):
    """
    Type for variable-length array of homogeneous elements.
    """
    kind = TypeKind.list

    def __init__(self, elem_ty: Type):
        self.elem_ty_ = elem_ty

    @property
    def elem_type(self) -> Optional['Type']:
        return self.elem_ty_

    def __eq__(self, other: Type):
        if other.kind == TypeKind.tuple:
            return other.__eq__(self)
        elif self.kind != other.kind:
            return False
        other = cast(ListType, other)
        return self.elem_ty_ == other.elem_ty_

    def __str__(self):
        return f'[{self.elem_ty_}]'


ValueType = Union[bool, int, float, str, tuple, list, DataType]

_type_funcs: Dict[t.Type, Callable[[ValueType], Type]] = {
    bool: lambda v: BOOL,
    int: lambda v: INT,
    float: lambda v: FLOAT,
    str: lambda v: STR,
    DataType: lambda v: DTYPE,
    tuple: lambda v: TupleType(*(type_py_value(f) for f in v)),
    list: lambda v: ListType(type_py_value(v[0])),
}


def type_py_value(v: ValueType) -> Type:
    """
    Find the corresponding type of Python value.
    :param v: Any acceptable Python object.
    :return: Type of `v`.
    """
    py_ty = type(v)
    if py_ty in _type_funcs:
        return _type_funcs[py_ty](v)
    else:
        raise TypeError(
            'Cannot type Python object of type {}'.format(
                util.cls_name(v))
        )


class TyVar(Type):
    """
    Type variable. This type is just for type inference and should not appear in user code.
    """
    kind = TypeKind.var

    def __str__(self):
        return '?'

    def __eq__(self, other):
        return self is other
