from enum import IntEnum, auto
from typing import Optional, List, cast

from ..expr import Expr, Const
from ..expr.array import Tuple
from ..expr.basic import Dummy
from ..expr.tensor import TensorKind
from ..expr.ty import Type, ValueType
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
    def defined(self) -> bool:
        raise NotImplemented

    def set_defined(self, expr: Expr):
        raise NotImplemented

    @property
    def expr(self) -> Expr:
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
        self.status_ = ValueStatus.UNDEFINED
        self.expr_: Optional[Expr] = None
        self.value_: Optional[ValueType] = None

    @property
    def defined(self) -> bool:
        return self.status_ != ValueStatus.UNDEFINED

    def set_defined(self, expr: Expr):
        self.status_ = ValueStatus.DEFINED
        self.expr_ = expr

    def set_solved(self, value: ValueType):
        self.status_ = ValueStatus.SOLVED
        self.value_ = value

    @property
    def expr(self) -> Expr:
        if self.status_ == ValueStatus.UNDEFINED:
            return Dummy()
        elif self.status_ == ValueStatus.DEFINED:
            return self.expr_
        elif self.status_ == ValueStatus.SOLVED:
            return Const(self.value_)

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
        self.expr_ = None
        self.nodes_: List[StoreNode] = []

    @property
    def len_defined(self):
        return self.len_.defined

    @property
    def expr_defined(self):
        return self.expr_ is not None

    @property
    def defined(self) -> bool:
        return self.expr_defined

    @property
    def len_solved(self):
        return self.len_.status_ == ValueStatus.SOLVED

    @property
    def elem_defined(self):
        return self.len_solved and (len(self.nodes_) == 0 or self.nodes_[0].defined)

    def set_len_defined(self, expr: Expr):
        self.len_.set_defined(expr)

    def set_expr_defined(self, expr: Expr):
        self.expr_ = expr

    def set_defined(self, expr: Expr):
        self.set_expr_defined(expr)

    def set_len_solved(self, length: int):
        assert self.expr_defined
        self.len_.set_solved(length)
        self.nodes_ = [StoreNode.create(self.expr_.type_.elem_type) for _ in range(length)]

    def set_elem_defined(self, tup: Tuple):
        if not self.len_solved:
            self.set_len_solved(len(tup.fields_))
        assert self.len_.value == len(tup.fields_)
        for (node, expr) in zip(self.nodes_, tup.fields_):
            node.set_defined(expr)

    @property
    def expr(self) -> Expr:
        if self.len_solved:
            return Tuple(*(n.expr for n in self.nodes_))
        else:
            return Dummy()

    @property
    def value(self) -> Optional[ValueType]:
        if not self.len_solved:
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

    def query_shape(self, kind: TensorKind, *ind: int):
        if kind == TensorKind.input:
            return self.query_in_shape(*ind)
        else:
            return self.query_out_shape(*ind)

    def query_in_shape(self, *ind: int):
        return self._query_node(self.in_shapes_, *ind)

    def query_out_shape(self, *ind: int):
        return self._query_node(self.out_shapes_, *ind)

    def query_dtype(self, kind: TensorKind, idx: int):
        if kind == TensorKind.input:
            return self.query_in_dtype(idx)
        else:
            return self.query_out_dtype(idx)

    def query_in_dtype(self, idx: int):
        return self._query_node(self.in_dtypes_, idx)

    def query_out_dtype(self, idx: int):
        return self._query_node(self.out_dtypes_, idx)

    @staticmethod
    def _query_node(node: StoreNode, *ind: int) -> Optional[StoreNode]:
        for idx in ind:
            if node.kind == NodeKind.SCALAR:
                raise RuntimeError('Cannot access scalar node by index.')
            arr_node = cast(ArrayNode, node)
            if not arr_node.len_solved:
                return None
            node = arr_node.nodes_[idx]
        return node
