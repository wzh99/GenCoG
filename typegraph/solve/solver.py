from typing import Dict, List, cast

from numpy.random import Generator

from .eval import PartialEval, EvalExpr
from .store import ValueStore, StoreNode, NodeKind, ValueStatus, ScalarNode, ArrayNode
from .valid import find_valid
from ..expr.array import Tuple
from ..expr.basic import Expr, ExprKind, Const, And, Var
from ..expr.fmt import print_expr
from ..expr.ty import TensorType, BOOL, INT, FLOAT
from ..expr.visitor import CopyExpr, StructuralEq
from ..spec import ConstraintSpec
from ..util import CodeBuffer


class SolveError(Exception):
    """
    The solver cannot solve constraint specification.
    """

    def __init__(self, solver: 'ConstraintSolver', msg: str):
        self.solver_ = solver
        self.msg_ = msg


class ConstraintSolver:
    """
    Solver of constraint specification.
    """

    def __init__(self, spec: ConstraintSpec, known: Dict[int, TensorType], rng: Generator):
        # Initialize value store and expression visitors
        self._spec = spec
        self._store = ValueStore(spec.attrs)
        self._known = known
        self._rng = rng
        self._partial = PartialEval(self._store)
        self._eval = EvalExpr(self._store)
        self._eq = StructuralEq()

        # Copy expressions to avoid polluting original specification
        cp = CopyExpr()
        self._in_ranks = cp.copy(spec.in_ranks)
        self._extra = [cp.copy(e) for e in spec.extra]
        shape_root = self._store.in_shapes_
        in_num = cp.copy(spec.in_num)
        shape_root.set_len_defined(in_num)
        shape_root.set_expr_defined(cp.copy(spec.in_shapes))
        dtype_root = self._store.in_dtypes_
        dtype_root.set_len_defined(in_num)
        dtype_root.set_expr_defined(cp.copy(spec.in_dtypes))

    def solve(self):
        # Solve attributes and inputs
        while True:
            if not self._solve_one_iter():
                break
        print(self._store)
        _print_extra(self._extra)

    def _solve_one_iter(self):
        # Solve attributes
        changed = False
        for _, node in self._store.attrs_:
            changed |= self._solve_node(node)

        # Solve inputs
        changed |= self._solve_shapes(self._store.in_shapes_)
        changed |= self._solve_dtypes(self._store.in_dtypes_)

        # Solve extra constraints
        changed |= self._solve_extra()

        # Solve with SMT
        self._solve_smt()

        return changed

    def _solve_shapes(self, root: ArrayNode) -> bool:
        # Solve number
        if not root.len_solved:
            return self._solve_len(root, by_elem=False)

        # Solve tensors
        if not root.elem_defined:
            # Partially evaluate ranks
            ranks = self._partial.transform(self._in_ranks)
            if ranks.kind != ExprKind.TUPLE:
                return False
            ranks = cast(Tuple, ranks)
            if root.len_.value != len(ranks.fields_):
                raise SolveError(
                    self,
                    f'Length of input rank array {len(ranks.fields_)} is not consistent with '
                    f'input number {root.len_.value}. '
                )

            # Define ranks for each input tensor
            for tensor, rank in zip(root.children_, ranks.fields_):
                tensor = cast(ArrayNode, tensor)
                tensor.set_len_defined(rank)

            # Partially evaluate shapes
            shapes = self._partial.transform(root.expr_)
            if shapes.kind != ExprKind.TUPLE:
                return False
            shapes = cast(Tuple, shapes)
            if root.len_.value != len(shapes.fields_):
                raise SolveError(
                    self,
                    f'Length of input shape array {len(shapes.fields_)} is not consistent with '
                    f'input number {root.len_.value}. '
                )

            # Define shapes for each input tensor
            root.set_elem_defined(shapes)
            return True

        # Solve shapes
        changed = False
        for t_idx, tensor in enumerate(root.children_):
            # Solve rank
            tensor = cast(ArrayNode, tensor)
            prev_solved = tensor.len_solved
            if t_idx in self._known:
                tensor.set_len_solved(self._known[t_idx].rank)
            changed |= self._solve_len(tensor, by_elem=False)
            changed |= prev_solved != tensor.len_solved
            if not tensor.len_solved:
                continue

            # Partially evaluate dimensions
            if not tensor.elem_defined:
                shape = self._partial.transform(tensor.expr_)
                if shape.kind != ExprKind.TUPLE:
                    continue
                shape = cast(Tuple, shape)
                if tensor.len_.value != len(shape.fields_):
                    raise SolveError(
                        self,
                        f'Length of input shape {len(shape.fields_)} for tensor {t_idx} is not '
                        f'consistent with rank {tensor.len_.value}. '
                    )
                tensor.set_elem_defined(shape)
                changed = True

            # Solve dimensions
            for d_idx, dim in enumerate(tensor.children_):
                dim = cast(ScalarNode, dim)
                prev_solved = dim.solved
                if t_idx in self._known:
                    known_shape = self._known[t_idx].shape_
                    dim.set_solved(known_shape[d_idx])
                changed |= self._solve_scalar(cast(ScalarNode, dim))
                changed |= prev_solved != dim.solved

        return changed

    def _solve_dtypes(self, root: ArrayNode) -> bool:
        # Solve number
        if not root.len_solved:
            return self._solve_len(root, by_elem=False)

        # Solve tensors
        if not root.elem_defined:
            # Partially evaluate data types
            dtypes = self._partial.transform(root.expr_)
            if dtypes.kind != ExprKind.TUPLE:
                return False
            dtypes = cast(Tuple, dtypes)
            root.set_elem_defined(dtypes)
            if root.len_.value != len(dtypes.fields_):
                raise SolveError(
                    self,
                    f'Length of input rank array {len(dtypes.fields_)} is not consistent with '
                    f'input number {root.len_.value}. '
                )

            # Define data type for each tensor
            root.set_elem_defined(dtypes)
            return True

        # Solve data types
        changed = False
        for t_idx, dtype in enumerate(root.children_):
            dtype = cast(ScalarNode, dtype)
            prev_solved = dtype.solved
            if t_idx in self._known:
                dtype.set_solved(self._known[t_idx].dtype_)
            changed |= self._solve_scalar(dtype)
            changed |= prev_solved != dtype.solved

        return changed

    def _solve_extra(self) -> bool:
        new_extra = []
        changed = False

        for e in self._extra:
            post = self._partial.transform(e)
            if post.kind == ExprKind.CONST:
                const = cast(Const, post)
                if const.val_ is False:
                    raise SolveError(
                        self, 'Extra constraint is not satisfiable.'
                    )
            elif post.kind == ExprKind.AND:
                and_e = cast(And, post)
                new_extra.extend(and_e.clauses_)
                changed = True
            else:
                changed |= not self._eq.visit(e, post)
                new_extra.append(post)

        self._extra = new_extra
        return changed

    def _solve_smt(self):
        # Find all valid variables and constraints with union-find
        union = find_valid(self._store, self._extra)
        all_valid = list(union.all_valid())
        if len(all_valid) == 0:
            return False

        # Sample variables that are not bounded by other constraints
        for e in all_valid:
            if e.kind != ExprKind.VAR:
                continue
            if union.has_use(e):
                continue  # if it has use, it is bounded by other constraints
            self._try_sample(cast(Var, e))
            pass
        return True

    def _try_sample(self, var: Var):
        # Directly sample boolean values
        if var.type_ == BOOL:
            v = bool(self._rng.integers(2))
            self._store.set_var_solved(var, v)

        # Get range for numeric values
        if var.ran_ is None:
            return
        ran = var.ran_
        if ran.begin_ is None or ran.begin_.kind != ExprKind.CONST:
            return
        low = cast(Const, ran.begin_).val_
        if ran.end_ is None or ran.end_.kind != ExprKind.CONST:
            return
        high = cast(Const, ran.end_).val_

        # Sample numeric values
        if var.type_ == INT:
            v = int(self._rng.integers(low=low, high=high))
            self._store.set_var_solved(var, v)
        elif var.type_ == FLOAT:
            v = float(self._rng.uniform(low=low, high=high))
            self._store.set_var_solved(var, v)

    def _solve_node(self, node: StoreNode) -> bool:
        if node.kind == NodeKind.SCALAR:
            return self._solve_scalar(cast(ScalarNode, node))
        else:
            return self._solve_array(cast(ArrayNode, node))

    def _solve_scalar(self, node: ScalarNode) -> bool:
        if node.status_ == ValueStatus.UNDEFINED:
            return False  # undefined or solved node cannot be processed
        post = self._partial.transform(node.expr_)
        if post.kind == ExprKind.CONST:
            prev_solved = node.solved
            node.set_solved(cast(Const, post).val_)
            return not prev_solved
        else:
            pre = node.expr_
            node.expr_ = post
            return not self._eq.visit(pre, post)

    def _solve_array(self, node: ArrayNode) -> bool:
        # Solve array length
        changed = False
        if not node.len_solved:  # solve length first
            if not self._solve_len(node):
                return False
            changed = True  # length is solved this time

        # Solve element expression
        if node.expr_defined and not node.elem_defined:
            changed |= self._solve_arr_elem(node)

        # Recursively solve children nodes
        for child in node.children_:
            changed |= self._solve_node(child)

        return changed

    def _solve_len(self, node: ArrayNode, by_elem: bool = True) -> bool:
        # Solve length directly
        if node.len_defined and self._solve_scalar(node.len_):
            node.set_len_solved(node.len_.value)
            return True

        # Solve length through partial evaluation of array expression
        if node.expr_defined and by_elem:
            return self._solve_arr_elem(node)

        return False

    def _solve_arr_elem(self, node: ArrayNode) -> bool:
        tup = self._partial.transform(node.expr_)
        if tup.kind != ExprKind.TUPLE:
            return False
        node.set_elem_defined(cast(Tuple, tup))
        return True


def _print_extra(extra: List[Expr]):
    buf = CodeBuffer()
    buf.write_pos_multi(
        map(lambda e: lambda: print_expr(e, buf, []), extra),
        prefix='[', suffix=']'
    )
    print(buf)
