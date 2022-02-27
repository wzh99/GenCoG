from typing import Iterable, List, cast, Optional, Dict

import numpy as np
from numpy.random import Generator

from .base import Input, Operation, Value, Graph, Output
from .lookup import OpLookup, ValueLookup
from ..config import params
from ..expr.ty import float_dtypes, common_dtypes
from ..solve import TensorType, TypeSolver, SolveError, OpTypeInfo
from ..solve.store import ArrayNode, ScalarNode
from ..spec import Op, TypeSpec, int_expr_choices, expr_choices, max_in_num, max_rank, max_dim
from ..util import inc_cnt

max_opr_num: int = params['graph.max_opr_num']
opr_trials: int = params['graph.opr_trials']
use_penal: float = params['graph.use_penal']


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
        inputs = []
        oprs = []
        value_lu = ValueLookup()

        # Generate initial input
        init_in = self._gen_input()
        inputs.append(init_in)
        value_lu.add(init_in.value_)

        # Iteratively construct computation graph
        while len(oprs) < max_opr_num:
            # Choose a value
            value = self._sample_value(list(value_lu.values), {})

            # Choose an operator whose first input matches this value
            op = self._sample_op(value)

            # Generate operation vertex
            opr = self._gen_opr(op, value, value_lu, inputs)
            if opr is None:
                continue
            else:
                oprs.append(opr)

        # Create final graph
        outputs = [Output(v) for v in value_lu.values if len(v.uses_) == 0]
        graph = Graph(inputs, outputs, oprs)

        return graph

    def _gen_input(self):
        rank = self._rng.integers(low=2, high=max_rank, endpoint=True)
        shape = cast(List[int],
                     self._rng.integers(low=1, high=max_dim, size=rank, endpoint=True).tolist())
        dtype = self._rng.choice(float_dtypes)
        return Input(TensorType(shape, dtype), False)

    def _sample_value(self, values: List[Value], add_cnt: Dict[Value, int]):
        num_uses = [len(v.uses_) + add_cnt.get(v, 0) for v in values]
        scores = softmax(-use_penal * np.array(num_uses, dtype='float32'))
        return self._rng.choice(values, p=scores)

    def _sample_op(self, value: Value) -> Op:
        ops = list(self._ops.by_first_type(value.type_))
        # TODO: Design fusion-aware heuristics to choose operators
        return self._rng.choice(ops)

    def _gen_opr(self, op: Op, fst_in: Value, value_lu: ValueLookup, graph_inputs: List[Input]) \
            -> Optional[Operation]:
        spec = op.spec
        if spec.is_variadic:
            return self._gen_variadic_opr(op, spec, fst_in, value_lu, graph_inputs)
        else:
            for _ in range(opr_trials):
                opr = self._gen_normal_opr(op, spec, fst_in, value_lu, graph_inputs)
                if opr is not None:
                    return opr
            return None

    def _gen_normal_opr(self, op: Op, spec: TypeSpec, fst_in: Value, value_lu: ValueLookup,
                        graph_inputs: List[Input]) -> Optional[Operation]:
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
        matched_cnt = {fst_in: 1}
        for t_idx in range(1, num):
            # Skip if this input tensor is a parameter
            if t_idx in op.params_:
                continue

            # Sample values in lookup table with matching shape and data type
            shape_node = cast(ArrayNode, shapes_node.children_[t_idx])
            dtype_node = cast(ScalarNode, dtypes_node.children_[t_idx])
            value = self._sample_match_value(value_lu, shape_node, dtype_node, matched_cnt)
            if value is None:
                continue
            matched[t_idx] = value
            inc_cnt(matched_cnt, value)

        # Perform complete solving to check whether inputs satisfy type constraints
        known = dict((i, v.type_) for i, v in matched.items())
        solver = TypeSolver(spec, known, self._rng)
        try:
            info = solver.solve()
        except SolveError:
            return None

        # Create operation
        return self._create_opr(op, info, matched, value_lu, graph_inputs)

    def _gen_variadic_opr(self, op: Op, spec: TypeSpec, fst_in: Value, value_lu: ValueLookup,
                          graph_inputs: List[Input]) -> Optional[Operation]:
        # Create solver and manually set input number
        solver = TypeSolver(spec, {0: fst_in.type_}, self._rng)
        store = solver.store_
        store.in_shapes_.set_len_solved(2)
        store.in_dtypes_.set_len_solved(2)

        # Perform initial solving
        try:
            solver.solve_initial()
        except SolveError:
            return None
        shape_node = store.in_shapes_.children_[1]
        dtype_node = store.in_dtypes_.children_[1]

        # Try finding matching values
        matched = {0: fst_in}
        matched_cnt = {fst_in: 1}
        info: Optional[OpTypeInfo] = None
        while len(matched) < max_in_num:
            # Find another value matching the pattern
            found = False
            idx = len(matched)

            for _ in range(opr_trials):
                # Sample matched value
                value = self._sample_match_value(value_lu, shape_node, dtype_node, matched_cnt)
                if value is None:
                    break

                # Check if the new inputs satisfy type constraints
                known = dict((i, v.type_) for i, v in matched.items())
                known[idx] = value.type_
                solver = TypeSolver(spec, known, self._rng)
                solver.store_.in_shapes_.set_len_solved(len(known))
                solver.store_.in_dtypes_.set_len_solved(len(known))
                try:
                    info = solver.solve()
                except SolveError:
                    continue

                # Add this value to matched set
                matched[idx] = value
                inc_cnt(matched_cnt, value)
                break

            # Stop finding if no values is found in this iteration
            if not found:
                break

        # Create operation
        if info is None:
            return None
        return self._create_opr(op, info, matched, value_lu, graph_inputs)

    @staticmethod
    def _create_opr(op: Op, info: OpTypeInfo, matched: Dict[int, Value],
                    value_lu: ValueLookup, graph_inputs: List[Input]):
        # Create operation
        inputs = []
        for idx in range(len(info.in_types_)):
            if idx in matched:
                inputs.append(matched[idx])
            else:
                param = Input(info.in_types_[idx], True)
                inputs.append(param.value_)
                graph_inputs.append(param)

        outputs = [Value(ty) for ty in info.out_types_]
        opr = Operation(op, info.attrs_, inputs, outputs)

        # Add output values to value lookup table
        for idx, out in enumerate(outputs):
            if idx not in op.ignored_:
                value_lu.add(out)

        return opr

    def _sample_match_value(self, value_lu: ValueLookup, shape: ArrayNode, dtype: ScalarNode,
                            add_cnt: Dict[Value, int]) -> Optional[Value]:
        # Compute choices of ranks and data types
        rank_choices = int_expr_choices(shape.len_.expr, 2, max_rank + 1)
        dtype_choices = expr_choices(dtype.expr, common_dtypes)

        # Query value lookup table
        matches = list(filter(lambda v: self._match_shape(shape, v),
                              value_lu.by_choices(rank_choices, dtype_choices)))
        return None if len(matches) == 0 else self._sample_value(matches, add_cnt)

    @staticmethod
    def _match_shape(shape: ArrayNode, value: Value):
        if not shape.elem_defined:
            return True
        assert shape.len_.value == value.type_.rank
        for dim_node, dim in zip(shape.children_, value.type_.shape_):
            if not dim_node.defined:
                return True
            if dim not in int_expr_choices(dim_node.expr, 1, max_dim + 1):
                return False
        return True
