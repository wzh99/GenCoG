from typing import Iterable, List, cast, Optional

import numpy as np
from numpy.random import Generator

from .base import Input, Operation, Value
from .lookup import OpLookup, ValueLookup
from ..config import config
from ..expr.ty import float_dtypes, common_dtypes
from ..solve import TensorType, TypeSolver, SolveError
from ..solve.store import ArrayNode, ScalarNode
from ..spec import Op, max_rank, max_dim, int_expr_choices, expr_choices

max_opr_num: int = config['graph.max_opr_num']
use_penal: float = config['graph.use_penal']


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    exp = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp / np.sum(exp, axis=axis, keepdims=True)


class GraphGenerator:
    """
    Type-directed computation graph generation.
    """

    def __init__(self, ops: Iterable[Op], rng: Generator):
        self._ops = OpLookup(ops)
        self._rng = rng

    def generate(self):
        # Initialization
        value_lu = ValueLookup()
        value_lu.add(self._gen_input().value_)
        oprs: List[Operation] = []

        # Iteratively construct computation graph
        while len(oprs) <= max_opr_num:
            # Choose a value
            value = self._sample_value(list(value_lu.values))

            # Choose an operator whose first input matches this value
            op = self._sample_op(value)

            # Resolve remaining input and output values of the operation
            opr = self._resolve_op(op, value, value_lu)
            if opr is None:
                continue
            else:
                oprs.append(opr)

            # Add operation to existing graph
            pass

        # Create final graph
        pass

    def _gen_input(self):
        rank = self._rng.integers(low=2, high=max_rank, endpoint=True)
        shape = cast(List[int],
                     self._rng.integers(low=1, high=max_dim, size=rank, endpoint=True).tolist())
        dtype = self._rng.choice(float_dtypes)
        return Input(TensorType(shape, dtype), False)

    def _sample_value(self, values: List[Value]):
        num_uses = [len(v.uses_) for v in values]
        scores = softmax(-use_penal * np.array(num_uses, dtype='float32'))
        return self._rng.choice(values, p=scores)

    def _sample_op(self, value: Value) -> Op:
        ops = list(self._ops.by_first_type(value.type_))
        # TODO: Design fusion-aware heuristics to choose operators
        return self._rng.choice(ops)

    def _resolve_op(self, op: Op, fst_in: Value, value_lu: ValueLookup) -> Optional[Operation]:
        # Handle variadic operator
        spec = op.spec
        if spec.is_variadic:
            # TODO: Resolve variadic operators
            return None

        # Perform initial solving
        solver = TypeSolver(spec, {0: fst_in.type_}, self._rng)
        try:
            solver.solve_initial()
        except SolveError:
            return None

        # Check nodes in value store
        store = solver.store_
        shapes_node = store.in_shapes_
        dtypes_node = store.in_dtypes_
        assert shapes_node.len_.value is not None
        assert shapes_node.len_.value == dtypes_node.len_.value
        num = shapes_node.len_.value

        # Matching existing values with remaining inputs of this operator
        matched = {0: fst_in}
        for t_idx in range(1, num):
            # Skip if this input tensor is a parameter
            if t_idx in op.params_:
                continue

            # Sample values in lookup table with matching shape and data type
            shape_node = cast(ArrayNode, shapes_node.children_[0])
            dtype_node = cast(ScalarNode, dtypes_node.children_[0])
            value = self._sample_matched_value(value_lu, shape_node, dtype_node)
            if value is None:
                continue
            matched[t_idx] = value

        # Perform complete solving to check whether inputs satisfy type constraints
        known = dict((i, v.type_) for i, v in matched.items())
        solver = TypeSolver(spec, known, self._rng)
        try:
            info = solver.solve()
        except SolveError:
            return None

        # Create input and output values
        ins = [matched[i] if i in matched else Input(info.in_types_[i], True).value_
               for i in range(len(info.in_types_))]
        outs = [Value(ty) for ty in info.out_types_]
        opr = Operation(op, info.attrs_, ins, outs)

        return opr

    def _sample_matched_value(self, value_lu: ValueLookup, shape: ArrayNode, dtype: ScalarNode) \
            -> Optional[Value]:
        # Compute choices of ranks and data types
        rank_choices = int_expr_choices(shape.len_.expr, 2, max_rank + 1)
        dtype_choices = expr_choices(dtype.expr, common_dtypes)

        # Query value lookup table
        matched = list(value_lu.by_choices(rank_choices, dtype_choices))
        return None if len(matched) == 0 else self._sample_value(matched)
