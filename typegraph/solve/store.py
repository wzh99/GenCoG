from enum import IntEnum, auto
from typing import Optional, List, cast

from ..expr import Expr
from ..expr.ty import Type, ValueType, ListType, INT, DTYPE
from ..spec import Attr


class NodeKind(IntEnum):
    SCALAR = auto()
    ARRAY = auto()


class ValueStatus(IntEnum):
    UNDEFINED = auto()  # Expression is undefined.
    DEFINED = auto()  # Expression is defined. Exact value is not worked out.
    SOLVED = auto()  # Exact value is worked out.


class StoreNode:
    """
    Node in a value store.
    """
    kind: NodeKind

    @staticmethod
    def create(ty: Type) -> 'StoreNode':
        return ScalarNode() if ty.is_scalar else ArrayNode()

    @property
    def status(self) -> ValueStatus:
        raise NotImplemented

    @property
    def value(self) -> Optional[ValueType]:
        raise NotImplemented


class ScalarNode(StoreNode):
    """
    Store status of a single scalar value.
    """
    kind = NodeKind.SCALAR

    def __init__(self):
        self._status = ValueStatus.UNDEFINED
        self.expr_: Optional[Expr] = None
        self.value_: Optional[ValueType] = None

    def set_defined(self, expr: Expr):
        self._status = ValueStatus.DEFINED
        self.expr_ = expr

    def set_solved(self, value: ValueType):
        self._status = ValueStatus.SOLVED
        self.value_ = value

    @property
    def status(self) -> ValueStatus:
        return self._status

    @property
    def value(self) -> Optional[ValueType]:
        return self.value_


class ArrayNode(StoreNode):
    """
    Store nodes of elements in an array.
    """
    kind = NodeKind.ARRAY

    def __init__(self):
        self.len_ = ScalarNode()
        self.nodes_: List[StoreNode] = []

    def set_defined(self, len_expr: Expr):
        self.len_.set_defined(len_expr)

    def set_solved(self, len_val: ValueType, elem_ty: Type):
        assert type(len_val) is int
        len_val = cast(int, len_val)
        self.len_.set_solved(len_val)
        self.nodes_ = [StoreNode.create(elem_ty) for _ in range(len_val)]

    @property
    def status(self) -> ValueStatus:
        return self.len_.status

    @property
    def value(self) -> Optional[ValueType]:
        if self.status != ValueStatus.SOLVED:
            return None
        return [node.value for node in self.nodes_]  # some elements can be None


class ValueStore:
    """
    Store solution status for each scalar value in layered fashion.
    """

    def __init__(self, attrs: List[Attr]):
        self.attrs_ = dict((a.name_, StoreNode.create(a.expr_.type_)) for a in attrs)
        self.in_dtypes_ = ArrayNode()
        self.in_shapes_ = ArrayNode()
        self.out_dtypes_ = ArrayNode()
        self.out_shapes_ = ArrayNode()

    def query_attr(self, name: str, *ind: int):
        return self._query_node(self.attrs_[name], *ind)

    def query_in_shape(self, *ind: int):
        return self._query_node(self.in_shapes_, *ind)

    def query_in_dtype(self, *ind: int):
        return self._query_node(self.in_dtypes_, *ind)

    def query_out_shape(self, *ind: int):
        return self._query_node(self.out_shapes_, *ind)

    def query_out_dtype(self, *ind: int):
        return self._query_node(self.out_dtypes_, *ind)

    @staticmethod
    def _query_node(node: StoreNode, *ind: int) -> Optional[StoreNode]:
        for idx in ind:
            if node.kind == NodeKind.SCALAR:
                raise RuntimeError('Cannot access scalar node by index.')
            arr_node = cast(ArrayNode, node)
            if arr_node.status != ValueStatus.SOLVED:
                return None
            node = arr_node.nodes_[idx]
        return node

    def define_in_num(self, expr: Expr):
        self.in_shapes_.set_defined(expr)
        self.in_dtypes_.set_defined(expr)

    def set_in_num(self, num: int):
        self.in_shapes_.set_solved(num, ListType(INT))
        self.in_dtypes_.set_solved(num, DTYPE)

    def define_out_num(self, expr: Expr):
        self.out_shapes_.set_defined(expr)
        self.out_dtypes_.set_defined(expr)

    def set_out_num(self, num: int):
        self.out_shapes_.set_solved(num, ListType(INT))
        self.out_dtypes_.set_solved(num, DTYPE)
