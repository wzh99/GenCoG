from enum import IntEnum, auto
from typing import Optional, List, TypeVar, Generic, Callable, Dict, Any, cast

from ..expr import Expr, Const
from ..expr.array import Tuple
from ..expr.basic import ExprKind, Dummy
from ..expr.visitor import CopyExpr
from ..expr.tensor import TensorKind
from ..expr.ty import Type, ValueType
from ..expr.fmt import print_expr
from ..spec import Attr
from ..util import CodeBuffer, cls_name


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
    def create_undefined(ty: Type) -> 'StoreNode':
        return ScalarNode() if ty.is_scalar else ArrayNode()

    @staticmethod
    def create_defined(expr: Expr) -> 'StoreNode':
        return ScalarNode(expr=expr) if expr.type_.is_scalar else ArrayNode(expr=expr)

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

    def print(self, buf: CodeBuffer):
        raise NotImplemented


class ScalarNode(StoreNode):
    """
    Store status of a single scalar value.
    """
    kind = NodeKind.SCALAR

    def __init__(self, expr: Optional[Expr] = None):
        self.status_ = ValueStatus.UNDEFINED
        self.expr_: Optional[Expr] = None
        if expr is not None:
            self.set_defined(expr)
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

    def __init__(self, expr: Optional[Expr] = None):
        self.len_ = ScalarNode()
        self.expr_ = None
        self.children_: List[StoreNode] = []
        if expr is not None:
            self.set_expr_defined(expr)

    @property
    def len_defined(self):
        return self.len_.defined

    @property
    def expr_defined(self):
        return self.expr_ is not None

    @property
    def defined(self) -> bool:
        return self.expr_defined and self.len_defined

    @property
    def len_solved(self):
        return self.len_.status_ == ValueStatus.SOLVED

    @property
    def elem_defined(self):
        return self.len_solved and all(map(lambda c: c.defined, self.children_))

    def set_len_defined(self, expr: Expr):
        self.len_.set_defined(expr)
        if expr.kind == ExprKind.CONST and self.expr_defined:
            self.set_len_solved(cast(Const, expr).val_)

    def set_expr_defined(self, expr: Expr):
        self.expr_ = expr
        if expr.kind == ExprKind.TUPLE:
            self.set_elem_defined(cast(Tuple, expr))

    def set_defined(self, expr: Expr):
        self.set_expr_defined(expr)

    def set_len_solved(self, length: int):
        assert self.expr_defined
        self.len_.set_solved(length)
        self.children_ = [StoreNode.create_undefined(self.expr_.type_.elem_type)
                          for _ in range(length)]

    def set_elem_defined(self, tup: Tuple):
        if not self.len_solved:
            self.set_len_solved(len(tup.fields_))
        assert self.len_.value == len(tup.fields_)
        for (node, expr) in zip(self.children_, tup.fields_):
            node.set_defined(expr)

    @property
    def expr(self) -> Expr:
        if self.len_solved:
            return Tuple(*(n.expr for n in self.children_))
        else:
            return Dummy()

    @property
    def value(self) -> Optional[ValueType]:
        if not self.len_solved:
            return None
        return [node.value for node in self.children_]  # some elements can be None

    def print(self, buf: CodeBuffer):
        buf.write(cls_name(self))
        items = [('len', lambda: self.len_.print(buf))]
        if self.expr_ is not None:
            items.append(('expr', lambda: print_expr(self.expr_, buf, [])))
        items.append(
            ('children', lambda: buf.write_pos_multi(
                list(map(lambda n: lambda: n.print(buf), self.children_)),
                prefix='[', suffix=']'
            ))
        )
        buf.write_named_multi(items)


class ValueStore:
    """
    Store solution status for each scalar value in layered fashion.
    """

    def __init__(self, attrs: List[Attr]):
        cp = CopyExpr()
        self.attrs_ = list((a.name_, StoreNode.create_defined(cp.copy(a.expr_))) for a in attrs)
        self._attr_dict = dict(self.attrs_)
        self.in_shapes_ = ArrayNode()
        self.in_dtypes_ = ArrayNode()
        self.out_shapes_ = ArrayNode()
        self.out_dtypes_ = ArrayNode()

    def query_attr(self, name: str, *ind: int):
        return self._query_node(self._attr_dict[name], *ind)

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
            if idx >= len(arr_node.children_):
                return None
            node = arr_node.children_[idx]
        return node

    def __str__(self):
        printer = StorePrinter()
        buf = CodeBuffer()
        buf.write(cls_name(self))
        buf.write_named_multi([
            ('attrs', lambda: buf.write_named_multi(
                list(map(lambda p: (p[0], lambda: printer.visit(p[1], buf)), self.attrs_)),
                prefix='[', suffix=']'
            )),
            ('in_shapes', lambda: printer.visit(self.in_shapes_, buf)),
            ('in_dtypes', lambda: printer.visit(self.in_dtypes_, buf)),
            ('out_shapes', lambda: printer.visit(self.out_shapes_, buf)),
            ('out_dtypes', lambda: printer.visit(self.out_dtypes_, buf))
        ])
        return str(buf)


A = TypeVar('A')
R = TypeVar('R')


class StoreVisitor(Generic[A, R]):
    def __init__(self):
        self._methods: Dict[NodeKind, Callable[[Any, A], R]] = {
            NodeKind.SCALAR: self.visit_scalar,
            NodeKind.ARRAY: self.visit_array,
        }

    def visit(self, node: StoreNode, arg: A) -> R:
        return self._methods[node.kind](node, arg)

    def visit_scalar(self, node: ScalarNode, arg: A) -> R:
        pass

    def visit_array(self, node: ArrayNode, arg: A) -> R:
        pass


class StorePrinter(StoreVisitor[CodeBuffer, None]):
    def visit_scalar(self, node: ScalarNode, buf: CodeBuffer):
        buf.write(cls_name(node))
        items = [('status', lambda: buf.write(node.status_.name))]
        if node.expr_ is not None:
            items.append(('expr', lambda: print_expr(node.expr_, buf, [])))
        if node.value_ is not None:
            items.append(('value', lambda: buf.write(str(node.value_))))
        buf.write_named_multi(items)

    def visit_array(self, node: ArrayNode, buf: CodeBuffer):
        buf.write(cls_name(node))
        items = [('len', lambda: self.visit(node.len_, buf))]
        if node.expr_ is not None:
            items.append(('expr', lambda: print_expr(node.expr_, buf, [])))
        items.append(
            ('children', lambda: buf.write_pos_multi(
                list(map(lambda n: lambda: self.visit(n, buf), node.children_)),
                prefix='[', suffix=']'
            ))
        )
        buf.write_named_multi(items)
