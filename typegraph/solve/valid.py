import sys
from typing import List, Dict, Callable, Iterable, Any, cast

from .store import ValueStore, NodeKind, StoreNode, ScalarNode, ArrayNode, StoreVisitor, ValueStatus
from ..expr import Expr, Var
from ..expr.array import GetItem, Len, Concat, Slice, Map, ReduceArray, ReduceIndex, Filter, \
    InSet, Subset
from ..expr.basic import ExprKind, Const, ForAll, GetAttr, Dummy
from ..expr.tensor import Num, Rank, Shape, GetDType
from ..expr.visitor import ExprVisitor
from ..util import Ref


def validate(store: ValueStore, extra: List[Expr]) -> 'UnionFind':
    """
    Find all valid (solvable) constraint expressions.

    :param store: Value store indicating current progress of solving.
    :param extra: List of extra constraints on values.
    :return: Set of expressions that are truly solvable.
    """
    # Initialize union-find and visitors
    union = UnionFind()
    expr_find = ExprFinder(store, union)
    store_find = StoreFinder(expr_find)

    # Visit stores and constraints
    for _, node in store.attrs_:
        store_find.find(node)
    store_find.find(store.in_shapes_)
    store_find.find(store.in_dtypes_)
    for e in extra:
        expr_find.visit(e, e)

    # Return union-find result
    return union


class UnionFind:
    """
    Use union-find algorithm to find all valid constraints.

    All variables and constraints in a constraint specification can be partitioned into several
    equivalent classes, depending on whether they have dependencies or not. A constraint is
    unsolvable if any variable it refers to is undefined (which is unsolvable). A variable is
    unsolvable if any constraint referring to it is unsolvable.
    """

    def __init__(self):
        self._idx_map: Dict[Ref[Expr], int] = {}
        self._root: List[int] = []
        self._valid: List[bool] = []

    def union(self, d: Var, u: Expr):
        """
        Set two expressions with def-use relation in one equivalence class. We always set root of
        definition to be the root of the use. This is useful to check whether an expression has any
        use, at cost of union-find efficiency.

        :param d: Variable defined.
        :param u: Expression that *uses* this variable to defines a constraint.
        """
        di = self._find_expr(d)
        ui = self._find_expr(u)
        if di == ui:
            return
        self._root[di] = ui
        self._valid[ui] &= self._valid[di]

    def set_invalid(self, e: Expr):
        self._valid[self._find_expr(e)] = False

    def has_use(self, e: Expr):
        i = self._get_idx(e)
        return i != self._root[i]

    def all_valid(self) -> Iterable[Expr]:
        return map(lambda p: p[0].obj_,
                   filter(lambda p: self._valid[p[1]], self._idx_map.items()))

    def _find_expr(self, e: Expr):
        return self._find_idx(self._get_idx(e))

    def _find_idx(self, i: int):
        while i != self._root[i]:
            self._valid[self._root[i]] &= self._valid[i]
            self._root[i] = self._root[self._root[i]]
            i = self._root[i]
        return i

    def _get_idx(self, e: Expr):
        ref = Ref(e)
        if ref in self._idx_map:
            return self._idx_map[ref]
        idx = len(self._idx_map)
        self._idx_map[ref] = idx
        self._root.append(idx)
        self._valid.append(True)
        return idx


class StoreFinder(StoreVisitor[None, None]):
    def __init__(self, expr_find: 'ExprFinder'):
        super().__init__()
        self._find = expr_find

    def find(self, node: StoreNode):
        self.visit(node, None)

    def visit_scalar(self, node: ScalarNode, arg: None):
        if node.status_ != ValueStatus.DEFINED:
            return  # do not care undefined and solved nodes
        if node.expr_.kind != ExprKind.VAR:
            return  # non-variable expression cannot be solved for nodes in store
        self._find.visit(node.expr_, node.expr_)


class ExprFinder(ExprVisitor[Expr, None]):
    """
    Find all valid variables in a specification.
    """

    def __init__(self, store: ValueStore, union: 'UnionFind'):
        super().__init__()
        self._store = store
        self._union = union

    def visit_var(self, var: Var, root: Expr):
        if var.tmpl_:
            return
        self._union.union(var, root)
        if var.ran_ is not None:
            self.visit(var.ran_, var)

    def visit_forall(self, forall: ForAll, root: Expr):
        self._set_invalid(forall, root, super().visit_forall)

    def visit_attr(self, attr: GetAttr, root: Expr):
        self._set_invalid(attr, attr, super().visit_attr)
        node = self._store.query_attr(attr.name_)
        self._union_store(node, root, sys.maxsize)

    def visit_dummy(self, dum: Dummy, root: Expr):
        self._union.set_invalid(root)

    def visit_num(self, num: Num, root: Expr):
        self._union.set_invalid(root)
        self._union_store(self._store.query_shape(num.kind), root, 1)

    def visit_rank(self, rank: Rank, root: Expr):
        self._set_invalid(rank, root, super().visit_rank)
        if rank.index.kind == ExprKind.CONST:
            idx = cast(Const, rank.index).val_
            node = self._store.query_in_shape(idx)
            self._union_store(node, root, 1)
        else:
            node = self._store.in_shapes_
            self._union_store(node, root, 2, include_len=False)

    def visit_shape(self, shape: Shape, root: Expr):
        self._set_invalid(shape, root, super().visit_shape)
        if shape.index.kind == ExprKind.CONST:
            idx = cast(Const, shape.index).val_
            node = self._store.query_in_shape(idx)
            self._union_store(node, root, 2)
        else:
            node = self._store.in_shapes_
            self._union_store(node, root, 3, include_len=False)

    def visit_dtype(self, dtype: GetDType, root: Expr):
        self._set_invalid(dtype, root, super().visit_dtype)
        if dtype.index.kind == ExprKind.CONST:
            idx = cast(Const, dtype.index).val_
            node = self._store.query_in_dtype(idx)
            self._union_store(node, root, 1)
        else:
            node = self._store.in_dtypes_
            self._union_store(node, root, 2, include_len=False)

    def visit_getitem(self, getitem: GetItem, root: Expr):
        self._set_invalid(getitem, root, super().visit_getitem)

    def visit_len(self, ln: Len, root: Expr):
        self._set_invalid(ln, root, super().visit_len)

    def visit_concat(self, concat: Concat, root: Expr):
        self._set_invalid(concat, root, super().visit_concat)

    def visit_slice(self, slc: Slice, root: Expr):
        self._set_invalid(slc, root, super().visit_slice)

    def visit_map(self, m: Map, root: Expr):
        self._set_invalid(m, root, super().visit_map)

    def visit_reduce_array(self, red: ReduceArray, root: Expr):
        if red.arr_.kind != ExprKind.TUPLE:
            # non-tuple array cannot be translated
            self._union.set_invalid(root)
        super().visit_reduce_array(red, root)

    def visit_reduce_index(self, red: ReduceIndex, root: Expr):
        ran = red.ran_
        if ran.begin_.kind != ExprKind.CONST or ran.end_.kind != ExprKind.CONST:
            self._union.set_invalid(root)
        super().visit_reduce_index(red, root)

    def visit_filter(self, flt: Filter, root: Expr):
        self._set_invalid(flt, root, super().visit_filter)

    def visit_inset(self, inset: InSet, root: Expr):
        self._set_invalid(inset, root, super().visit_inset)

    def visit_subset(self, subset: Subset, root: Expr):
        self._set_invalid(subset, root, super().visit_subset)

    def _set_invalid(self, e: Expr, root: Expr, post_f: Callable[[Any, Any], Any]):
        self._union.set_invalid(root)
        post_f(e, root)

    def _union_store(self, node: StoreNode, root: Expr, level: int, include_len: bool = True):
        if level == 0:
            return
        if node.kind == NodeKind.SCALAR:
            node = cast(ScalarNode, node)
            if node.status_ != ValueStatus.DEFINED:
                return
            if node.expr_.kind != ExprKind.VAR:
                return
            self._union.union(cast(Var, node.expr_), root)
        elif node.kind == NodeKind.ARRAY:
            node = cast(ArrayNode, node)
            if node.expr_defined and not node.elem_defined:
                self.visit(node.expr_, root)
            if include_len:
                self._union_store(node.len_, root, level)
            for child in node.children_:
                self._union_store(child, root, level - 1)
