from typing import Dict, cast

from .eval import PartialEval, EvalExpr, EvalError
from .store import ValueStore, StoreNode, NodeKind, ValueStatus, ScalarNode, ArrayNode
from ..expr.array import Tuple
from ..expr.basic import ExprKind
from ..expr.visitor import CopyExpr
from ..expr.ty import TensorType
from ..spec import ConstraintSpec
from ..util import Ref


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

    def __init__(self, spec: ConstraintSpec, known: Dict[int, TensorType]):
        # Initialize value store and expression visitors
        self._spec = spec
        self._store = ValueStore(spec.attrs)
        self._known = known
        self._partial = PartialEval(self._store)
        self._eval = EvalExpr(self._store)

        # Copy expressions to avoid polluting original specification
        cp = CopyExpr()
        self._in_ranks = cp.copy(spec.in_ranks)
        self._extra = set(Ref(cp.copy(e)) for e in spec.extra)
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

    def _solve_one_iter(self):
        # Solve attributes
        changed = False
        for _, node in self._store.attrs_:
            changed |= self._solve_node(node)

        # Solve inputs
        changed |= self._solve_shapes(self._store.in_shapes_)
        changed |= self._solve_dtypes(self._store.in_dtypes_)
        print(self._store)
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
            for tensor, shape in zip(root.children_, shapes.fields_):
                tensor.set_defined(shape)

            return True

        # Solve shapes
        changed = False
        for t_idx, tensor in enumerate(root.children_):
            # Solve rank
            tensor = cast(ArrayNode, tensor)
            if not tensor.len_solved:
                if t_idx in self._known:
                    tensor.set_len_defined(self._known[t_idx].rank)
                    changed = True
                elif self._solve_len(tensor, by_elem=False):
                    changed = True
                else:
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
                for dim, expr in zip(tensor.children_, shape.fields_):
                    dim.set_defined(expr)
                changed = True

            # Solve dimensions
            for d_idx, dim in enumerate(tensor.children_):
                dim = cast(ScalarNode, dim)
                if dim.solved:
                    continue
                if t_idx in self._known:
                    known_shape = self._known[t_idx].shape_
                    if d_idx >= len(known_shape):
                        raise SolveError(
                            self,
                            f'Expect dimension {d_idx} from tensor {t_idx}, got only '
                            f'{len(known_shape)} dimensions.'
                        )
                    dim.set_solved(known_shape[d_idx])
                changed |= self._solve_scalar(cast(ScalarNode, dim))

        return changed

    def _solve_dtypes(self, root: ArrayNode) -> bool:
        # Solve number
        if not root.len_solved:
            self._solve_len(root, by_elem=False)

        # Solve tensors
        if not root.elem_defined:
            # Partially evaluate data types
            dtypes = self._partial.transform(root.expr_)
            if dtypes.kind != ExprKind.TUPLE:
                return False
            dtypes = cast(Tuple, dtypes)
            if root.len_.value != len(dtypes.fields_):
                raise SolveError(
                    self,
                    f'Length of input rank array {len(dtypes.fields_)} is not consistent with '
                    f'input number {root.len_.value}. '
                )

            # Define data type for each tensor
            for tensor, dtype in zip(root.children_, dtypes):
                tensor.set_defined(dtype)

            return True

        # Solve data types
        changed = False
        for t_idx, dtype in enumerate(root.children_):
            dtype = cast(ScalarNode, dtype)
            if dtype.solved:
                continue
            elif t_idx in self._known:
                dtype.set_solved(self._known[t_idx].dtype_)
            else:
                changed |= self._solve_scalar(cast(ScalarNode, dtype))

        return changed

    def _solve_node(self, node: StoreNode) -> bool:
        if node.kind == NodeKind.SCALAR:
            return self._solve_scalar(cast(ScalarNode, node))
        else:
            return self._solve_array(cast(ArrayNode, node))

    def _solve_scalar(self, node: ScalarNode) -> bool:
        if node.status_ != ValueStatus.DEFINED:
            return False  # undefined or solved node cannot be processes
        try:
            v = self._eval.evaluate(node.expr_)
            node.set_solved(v)
            return True
        except EvalError:
            return False

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
