from enum import IntEnum, auto
from typing import Optional, List, TypeVar, Generic, Callable, Dict, Any, cast

from ..expr import Expr, Const
from ..expr.array import Tuple
from ..expr.basic import ExprKind, Var, Dummy
from ..expr.visitor import CopyExpr
from ..expr.tensor import TensorKind
from ..expr.ty import Type, ValueType
from ..expr.fmt import print_expr
from ..spec import Attr
from ..util import CodeBuffer, Ref, cls_name


class NodeKind(IntEnum):
    SCALAR = auto()
    ARRAY = auto()


class ValueStatus(IntEnum):
    UNDEFINED = auto()  # Expression is undefined.
    DEFINED = auto()  # Expression is defined. Exact value is not worked out.
    SOLVED = auto()  # Exact value is worked out.


class StoreError(Exception):
    def __init__(self, store: 'ValueStore', node: Optional['StoreNode'], msg: str, ):
        self.store_ = store
        self.node_ = node
        self.msg_ = msg


class StoreNode:
    """
    Node in a value store.
    """
    kind: NodeKind

    def __init__(self, store: 'ValueStore'):
        self.store_ = store

    @staticmethod
    def create_undefined(store: 'ValueStore', ty: Type) -> 'StoreNode':
        return ScalarNode(store) if ty.is_scalar else ArrayNode(store)

    @staticmethod
    def create_defined(store: 'ValueStore', expr: Expr) -> 'StoreNode':
        return ScalarNode(store, expr=expr) if expr.type_.is_scalar \
            else ArrayNode(store, expr=expr)

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

    def __init__(self, store: 'ValueStore', expr: Optional[Expr] = None):
        super().__init__(store)
        self.status_ = ValueStatus.UNDEFINED
        self.expr_: Optional[Expr] = None
        if expr is not None:
            self.set_defined(expr)
        self.value_: Optional[ValueType] = None

    @property
    def defined(self) -> bool:
        return self.status_ != ValueStatus.UNDEFINED

    @property
    def solved(self):
        return self.status_ == ValueStatus.SOLVED

    def set_defined(self, expr: Expr):
        self.status_ = ValueStatus.DEFINED
        self.expr_ = expr

    def set_solved(self, value: ValueType):
        if self.solved and value != self.value_:
            raise StoreError(
                self.store_, self,
                f'Newly provided value {value} is not consistent with last solved value '
                f'{self.value_}.'
            )
        self.status_ = ValueStatus.SOLVED
        self.value_ = value
        if self.expr_ is not None and self.expr_.kind == ExprKind.VAR:
            var = cast(Var, self.expr_)
            self.store_.set_var_solved(var, value, self)

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

    def __init__(self, store: 'ValueStore', expr: Optional[Expr] = None):
        super().__init__(store)
        self.len_ = ScalarNode(store)
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
        self.children_ = [StoreNode.create_undefined(self.store_, self.expr_.type_.elem_type)
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


class ValueStore:
    """
    Store solution status for each scalar value in layered fashion.
    """

    def __init__(self, attrs: List[Attr]):
        cp = CopyExpr()
        self.attrs_ = list(
            (a.name_, StoreNode.create_defined(self, cp.copy(a.expr_))) for a in attrs)
        self._attr_dict = dict(self.attrs_)
        self._solved_var_: Dict[Ref[Var], ValueType] = {}
        self.in_shapes_ = ArrayNode(self)
        self.in_dtypes_ = ArrayNode(self)
        self.out_shapes_ = ArrayNode(self)
        self.out_dtypes_ = ArrayNode(self)

    def query_attr(self, name: str, *ind: int):
        if name not in self._attr_dict:
            raise StoreError(
                self, None, f'Attribute {name} not defined.'
            )
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

    def set_var_solved(self, var: Var, value: ValueType, node: StoreNode):
        var_ref = Ref(var)
        if var_ref not in self._solved_var_:
            self._solved_var_.update()
            self._solved_var_[var_ref] = value
        elif self._solved_var_[var_ref] != value:
            raise StoreError(
                self, node,
                f'Solved value {value} is not consistent with its previous result '
                f'{self._solved_var_[var_ref]}.'
            )

    def query_var(self, var: Var) -> Optional[ValueType]:
        return self._solved_var_.get(Ref(var))

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
