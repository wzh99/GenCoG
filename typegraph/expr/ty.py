import typing
from enum import IntEnum, auto

from .. import util


class TypeKind(IntEnum):
    """
    Simple RTTI mechanism for `Type`.
    """
    BOOL = auto()
    INT = auto()
    FLOAT = auto()
    STR = auto()
    DTYPE = auto()
    TUPLE = auto()
    LIST = auto()


class Type:
    """
    Base class for all expression types.
    """
    kind: TypeKind

    prim_kinds = [TypeKind.BOOL, TypeKind.INT, TypeKind.FLOAT, TypeKind.STR]

    def __eq__(self, other: 'Type'):
        """
        Compare structural equality of two type.
        :param other: The other type to be compared.
        :return: Whether the two types are structurally equal.
        """
        return self.kind == other.kind


class Bool(Type):
    """
    Boolean type.
    """
    kind = TypeKind.BOOL


class Int(Type):
    """
    Integer type.
    """
    kind = TypeKind.INT


class Float(Type):
    """
    Float type.
    """
    kind = TypeKind.FLOAT


class Str(Type):
    """
    String type.
    """
    kind = TypeKind.STR


class DType(Type):
    """
    Type for tensor data type.
    """
    kind = TypeKind.DTYPE


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

    def __eq__(self, other: 'DataType'):
        return self.code_ == other.code_ and self.bits_ == other.bits_

    def __str__(self):
        return self.code_.name + str(self.bits_)


class Tuple(Type):
    """
    Type for fixed-length array of possibly heterogeneous elements. A homogeneous tuple type can
    be cast to a list type.
    """
    kind = TypeKind.TUPLE

    def __init__(self, *field_ty: Type):
        if len(field_ty) == 0:
            raise ValueError(
                f'Expect at least one field type, got {len(field_ty)}.'
            )
        self.field_ty_ = field_ty
        self.is_homo_ = self._is_homo()

    def __eq__(self, other: Type):
        if other.kind == TypeKind.LIST:
            other = typing.cast(List, other)
            return self.is_homo_ and self.field_ty_[0] == other.elem_ty_
        elif self.kind != other.kind:
            return False
        other = typing.cast(Tuple, other)
        if len(self.field_ty_) != len(other.field_ty_):
            return False
        return all(map(lambda p: p[0] == p[1], zip(self.field_ty_, other.field_ty_)))

    def _is_homo(self):
        if len(self.field_ty_) == 1:
            return True
        return all(map(lambda t: t == self.field_ty_[0], self.field_ty_[1:]))


class List(Type):
    """
    Type for variable-length array of homogeneous elements.
    """
    kind = TypeKind.LIST

    def __init__(self, elem_ty: Type):
        self.elem_ty_ = elem_ty

    def __eq__(self, other: Type):
        if other.kind == TypeKind.TUPLE:
            return other.__eq__(self)
        elif self.kind != other.kind:
            return False
        other = typing.cast(List, other)
        return self.elem_ty_ == other.elem_ty_


ValueType = typing.Union[bool, int, float, str, tuple, list, DataType]


def type_py_value(v: ValueType) -> Type:
    """
    Find the corresponding type of Python value.
    :param v: Any acceptable Python object.
    :return: Type of `v`.
    """
    if v.__class__ == bool:
        return Bool()
    elif isinstance(v, int):
        return Int()
    elif isinstance(v, float):
        return Float()
    elif isinstance(v, str):
        return Str()
    elif isinstance(v, DataType):
        return DType()
    elif isinstance(v, tuple):
        return Tuple(*(type_py_value(f) for f in v))
    elif isinstance(v, list):
        return List(type_py_value(v[0]))
    else:
        raise TypeError(
            'Cannot type Python object of type {}'.format(
                util.cls_name(v))
        )
