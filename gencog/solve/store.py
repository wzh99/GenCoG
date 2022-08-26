import typing as t
from enum import IntEnum, auto
from typing import Optional, TypeVar, Generic, Callable, Dict, Any, Iterable, cast

from ..expr import Expr, Const
from ..expr.array import Tuple, List
from ..expr.basic import ExprKind, Var, Dummy
from ..expr.fmt import ExprPrinter, print_expr
from ..expr.tensor import TensorKind
from ..expr.ty import Type, ListType, ValueType, INT, DTYPE
from ..expr.visitor import CopyExpr
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
    def __init__(self, node: Optional['StoreNode'], msg: str):
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
        return ScalarNode(store, ty) if ty.is_scalar else ArrayNode(store, ty)

    @staticmethod
    def create_defined(store: 'ValueStore', expr: Expr) -> 'StoreNode':
        return ScalarNode(store, expr.type_, expr=expr) if expr.type_.is_scalar \
            else ArrayNode(store, expr.type_, expr=expr)

    @property
    def defined(self) -> bool:
        raise NotImplemented

    def set_defined(self, expr: Expr):
        raise NotImplemented

    @property
    def solved(self) -> bool:
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

    def __init__(self, store: 'ValueStore', ty: Type, expr: Optional[Expr] = None):
        super().__init__(store)
        self.status_ = ValueStatus.UNDEFINED
        self.type_ = ty
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
        if not self.defined:
            self.set_defined(Const(value))
        if self.solved and value != self.value_:
            raise StoreError(
                self,
                f'Newly provided value {value} is not consistent with last solved value '
                f'{self.value_}.'
            )
        self.status_ = ValueStatus.SOLVED
        self.value_ = value
        if self.expr_.kind == ExprKind.VAR:
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

    def __init__(self, store: 'ValueStore', ty: Type, expr: Optional[Expr] = None):
        super().__init__(store)
        self.len_ = ScalarNode(store, INT)
        self.type_ = ty
        self.expr_ = None
        self.children_: t.List[StoreNode] = []
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
    def solved(self) -> bool:
        if not self.len_solved:
            return False
        return all(c.solved for c in self.children_)

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
        elif expr.kind == ExprKind.LIST and not self.len_defined and \
                cast(List, expr).len_.kind == ExprKind.VAR:
            self.set_len_defined(cast(List, expr).len_)

    def set_defined(self, expr: Expr):
        self.set_expr_defined(expr)

    def set_len_solved(self, length: int):
        self.len_.set_solved(length)
        if len(self.children_) == 0:
            self.children_ = [StoreNode.create_undefined(self.store_, self.type_.elem_type)
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
        return tuple(node.value for node in self.children_)  # some elements can be None


class ValueStore:
    """
    Store solution status for each scalar value in layered fashion.
    """

    def __init__(self, attrs: Iterable[Attr]):
        cp = CopyExpr()
        self.attrs_ = list(
            (a.name_, StoreNode.create_defined(self, cp.copy(a.expr_))) for a in attrs)
        self._attr_dict = dict(self.attrs_)
        self._solved_var_: Dict[Ref[Var], ValueType] = {}
        self.in_shapes_ = ArrayNode(self, ListType(ListType(INT)))
        self.in_dtypes_ = ArrayNode(self, ListType(DTYPE))
        self.out_shapes_ = ArrayNode(self, ListType(ListType(INT)))
        self.out_dtypes_ = ArrayNode(self, ListType(DTYPE))

    def query_attr(self, name: str, *ind: int):
        if name not in self._attr_dict:
            raise StoreError(
                None, f'Attribute {name} not defined.'
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

    def set_var_solved(self, var: Var, value: ValueType, node: Optional[StoreNode] = None):
        var_ref = Ref(var)
        if var_ref not in self._solved_var_:
            self._solved_var_.update()
            self._solved_var_[var_ref] = value
        elif self._solved_var_[var_ref] != value:
            raise StoreError(
                node,
                f'Solved value {value} is not consistent with its previous result '
                f'{self._solved_var_[var_ref]}.'
            )

    def query_var(self, var: Var) -> Optional[ValueType]:
        return self._solved_var_.get(Ref(var))

    def print(self, buf: CodeBuffer):
        store_print = StorePrinter()
        expr_print = ExprPrinter(buf, [])
        buf.write(cls_name(self))
        buf.write_named_multi([
            ('attrs', lambda: buf.write_named_multi(
                list(map(lambda p: (p[0], lambda: store_print.visit(p[1], buf)), self.attrs_)),
                prefix='[', suffix=']'
            )),
            ('in_shapes', lambda: store_print.visit(self.in_shapes_, buf)),
            ('in_dtypes', lambda: store_print.visit(self.in_dtypes_, buf)),
            ('out_shapes', lambda: store_print.visit(self.out_shapes_, buf)),
            ('out_dtypes', lambda: store_print.visit(self.out_dtypes_, buf)),
            ('solved_vars', lambda: self._print_solved(expr_print, buf))
        ])

    def _print_solved(self, printer: 'ExprPrinter', buf: CodeBuffer):
        buf.writeln('{')
        with buf.indent():
            for ref, value in self._solved_var_.items():
                printer.visit_var(ref.obj_, None)
                buf.write('=')
                buf.write(str(value))
                buf.writeln(',')
        buf.write('}')

    def __repr__(self):
        buf = CodeBuffer()
        self.print(buf)
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
        self.visit(node.len_, arg)
        for child in node.children_:
            self.visit(child, arg)
        return


class StorePrinter(StoreVisitor[CodeBuffer, None]):
    def visit_scalar(self, node: ScalarNode, buf: CodeBuffer):
        buf.write(cls_name(node))
        items = [('status', lambda: buf.write(node.status_.name))]
        if node.expr_ is not None:
            items.append(('expr', lambda: print_expr(node.expr_, buf, [])))
        if node.value_ is not None:
            items.append(('value', lambda: buf.write(repr(node.value_))))
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
