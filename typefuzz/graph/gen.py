from typing import Iterable, List, cast

import numpy as np
from numpy.random import Generator

from .base import Input, Operation, Value
from .lookup import OpLookup, ValueLookup
from ..config import config
from ..expr.ty import float_dtypes
from ..solve import TensorType
from ..spec import Op, max_rank, max_dim

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
            value = self._sample_value(value_lu)

            # Choose an operator whose first input matches this value
            op = self._sample_op(value)

            # Resolve remaining input and output values of the operation
            if not self._resolve_op(op, value, value_lu):
                continue

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

    def _sample_value(self, value_lu: ValueLookup):
        values = list(value_lu.values)
        num_uses = [len(v.uses_) for v in values]
        scores = softmax(-use_penal * np.array(num_uses, dtype='float32'))
        return self._rng.choice(values, p=scores)

    def _sample_op(self, value: Value) -> Op:
        ops = list(self._ops.by_first_type(value.type_))
        # TODO: Design fusion-aware heuristics to choose operators
        return self._rng.choice(ops)

    def _resolve_op(self, op: Op, fst_in: Value, value_lu: ValueLookup) -> bool:
        pass
