import typing as t
from typing import Dict, Callable, Iterable, Any

from .store import ValueStore, StoreNode, ScalarNode, StoreVisitor, ValueStatus
from ..expr import Expr, Var, BOOL, INT, FLOAT
from ..expr.array import GetItem, Len, Concat, Slice, Map, ReduceArray, ReduceRange, \
    Filter, InSet, Subset, Perm, List
from ..expr.basic import ExprKind, ForAll, GetAttr, Dummy, Cond
from ..expr.tensor import Num, Rank, Shape, GetDType, LayoutMap, LayoutIndex
from ..expr.visitor import ExprVisitor
from ..util import Ref


def validate(store: ValueStore, extra: t.List[Expr]) -> 'UnionFind':
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
        self._root: t.List[int] = []
        self._valid: t.List[bool] = []

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
        items = sorted(self._idx_map.items(), key=lambda p: p[1])
        return map(lambda p: p[0].obj_,
                   filter(lambda p: self._valid[self._find_idx(p[1])], items))

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

    solvable_types = [BOOL, INT, FLOAT]

    def visit_var(self, var: Var, root: Expr):
        if var.tmpl_:
            return
        self._union.union(var, var)  # register this variable
        if var.type_ in self.solvable_types:
            # Only when the variable's type is solvable, we consider whether it is bounded by other
            # constraints. Otherwise, we consider this as free variable and use simple sampling to
            # solve it.
            self._union.union(var, root)
        elif var is not root:
            self._union.set_invalid(root)  # the constraint referring to it cannot be solved now
        if var.ran_ is not None:
            self.visit(var.ran_, var)
        if var.choices_ is not None:
            self.visit(var.choices_, var)

    def visit_forall(self, forall: ForAll, root: Expr):
        self._union.set_invalid(root)
        self.visit(forall.body_, root)

    def visit_cond(self, cond: Cond, root: Expr):
        self._union.set_invalid(root)
        self.visit(cond.tr_br_, root)
        self.visit(cond.fls_br_, root)

    def visit_attr(self, attr: GetAttr, root: Expr):
        self._set_invalid(attr, root, super().visit_attr)

    def visit_dummy(self, dum: Dummy, root: Expr):
        self._union.set_invalid(root)

    def visit_num(self, num: Num, root: Expr):
        self._union.set_invalid(root)

    def visit_rank(self, rank: Rank, root: Expr):
        self._set_invalid(rank, root, super().visit_rank)

    def visit_shape(self, shape: Shape, root: Expr):
        self._set_invalid(shape, root, super().visit_shape)

    def visit_dtype(self, dtype: GetDType, root: Expr):
        self._set_invalid(dtype, root, super().visit_dtype)

    def visit_layout_index(self, i: LayoutIndex, root: Expr):
        self._set_invalid(i, root, super().visit_layout_index)

    def visit_layout_map(self, m: LayoutMap, root: Expr):
        self._set_invalid(m, root, super().visit_layout_map)

    def visit_list(self, lst: List, root: Expr):
        self._union.set_invalid(root)
        self.visit(lst.body_, root)

    def visit_getitem(self, getitem: GetItem, root: Expr):
        self._union.set_invalid(root)
        self.visit(getitem.arr_, root)

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

    def visit_reduce_index(self, red: ReduceRange, root: Expr):
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

    def visit_perm(self, perm: Perm, root: Expr):
        self._set_invalid(perm, root, super().visit_perm)

    def _set_invalid(self, e: Expr, root: Expr, post_f: Callable[[Any, Any], Any]):
        self._union.set_invalid(root)
        post_f(e, root)
