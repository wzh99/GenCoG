from typing import List, Iterable, Dict, Callable, Any

from .store import ValueStore, StoreNode, ScalarNode, StoreVisitor, ValueStatus
from ..expr import Expr, Var
from ..expr.array import GetItem, Len, Concat, Slice, Map, ReduceArray, ReduceIndex, Filter, \
    InSet, Subset
from ..expr.basic import ExprKind, ForAll, Dummy
from ..expr.tensor import Num, Rank, Shape, GetDType
from ..expr.visitor import ExprVisitor
from ..util import Ref


def find_valid(store: ValueStore, extra: List[Expr]) -> 'UnionFind':
    """
    Find all valid (solvable) constraint expressions.

    :param store: Value store indicating current progress of solving.
    :param extra: List of extra constraints on values.
    :return: Set of expressions that are truly solvable.
    """
    # Initialize union-find and visitors
    union = UnionFind()
    expr_find = ExprVarFinder(union)
    store_find = StoreVarFinder(expr_find)

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

    def check(self, e: Expr):
        return self._valid[self._find_expr(e)]

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


class StoreVarFinder(StoreVisitor[None, None]):
    def __init__(self, expr_find: 'ExprVarFinder'):
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


class ExprVarFinder(ExprVisitor[Expr, None]):
    """
    Find all valid variables in a specification.
    """

    def __init__(self, union: 'UnionFind', ):
        super().__init__()
        self._union = union

    def visit_var(self, var: Var, use: Expr):
        self._union.union(var, use)
        if var.ran_ is not None:
            self.visit(var.ran_, var)

    def visit_forall(self, forall: ForAll, use: Expr):
        self._set_invalid(forall, use, super().visit_forall)

    def visit_dummy(self, dum: Dummy, use: Expr):
        self._union.set_invalid(use)

    def visit_num(self, num: Num, use: Expr):
        self._union.set_invalid(use)

    def visit_rank(self, rank: Rank, use: Expr):
        self._set_invalid(rank, use, super().visit_rank)

    def visit_shape(self, shape: Shape, use: Expr):
        self._set_invalid(shape, use, super().visit_shape)

    def visit_dtype(self, dtype: GetDType, use: Expr):
        self._set_invalid(dtype, use, super().visit_dtype)

    def visit_getitem(self, getitem: GetItem, use: Expr):
        self._set_invalid(getitem, use, super().visit_getitem)

    def visit_len(self, ln: Len, use: Expr):
        self._set_invalid(ln, use, super().visit_len)

    def visit_concat(self, concat: Concat, use: Expr):
        self._set_invalid(concat, use, super().visit_concat)

    def visit_slice(self, slc: Slice, use: Expr):
        self._set_invalid(slc, use, super().visit_slice)

    def visit_map(self, m: Map, use: Expr):
        self._set_invalid(m, use, super().visit_map)

    def visit_reduce_array(self, red: ReduceArray, use: Expr):
        if red.arr_.kind != ExprKind.TUPLE:
            # non-tuple array cannot be translated
            self._union.set_invalid(use)
        super().visit_reduce_array(red, use)

    def visit_reduce_index(self, red: ReduceIndex, use: Expr):
        ran = red.ran_
        if ran.begin_.kind != ExprKind.CONST or ran.end_.kind != ExprKind.CONST:
            self._union.set_invalid(use)
        super().visit_reduce_index(red, use)

    def visit_filter(self, flt: Filter, use: Expr):
        self._set_invalid(flt, use, super().visit_filter)

    def visit_inset(self, inset: InSet, use: Expr):
        self._set_invalid(inset, use, super().visit_inset)

    def visit_subset(self, subset: Subset, use: Expr):
        self._set_invalid(subset, use, super().visit_inset)

    def _set_invalid(self, e: Expr, use: Expr, post_f: Callable[[Any, Any], Any]):
        self._union.set_invalid(use)
        post_f(e, use)
