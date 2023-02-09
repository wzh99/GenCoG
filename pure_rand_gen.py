from argparse import Namespace, ArgumentParser
from sys import stdout
from typing import Iterable, List, cast, Tuple, Set

from numpy.random import PCG64, Generator
from tqdm import tqdm
from tvm import parser, TVMError

from gencog import Op, TypeSpec, DataType, Expr
from gencog.config import common_ops
from gencog.expr import array
from gencog.expr.basic import ExprKind
from gencog.expr.ty import ValueType
from gencog.graph import GraphGenerator, Input, Value, Operation, Output, Graph, print_relay
from gencog.solve import TensorType
from gencog.spec import int_expr_choices, max_in_num, max_rank, max_dim, expr_choices, OpRegistry

opr_num = 2


class PureRandomGenerator(GraphGenerator):
    def __init__(self, ops: Iterable[Op], rng: Generator):
        ops = list(ops)
        super().__init__(ops, rng)
        self._ops = ops

    def generate(self):
        # Initialization
        inputs = []
        oprs = []
        values = []

        # Generate initial input
        init_in = self._gen_input()
        inputs.append(init_in)
        values.append(init_in.value_)
        dtype = init_in.value_.type_.dtype_

        # Randomly generate vertices
        for _ in range(opr_num):
            # Choose an operator
            op = self._rng.choice(self._ops)
            spec = op.spec

            # Choose values as inputs
            op_inputs = self._sample_op_inputs(op, spec, dtype, values, inputs)

            # Randomly sample attributes
            attrs = self._sample_attrs(spec)

            # Create operation vertex
            output = Value(TensorType([], dtype))
            opr = Operation(op, attrs, op_inputs, [output])
            oprs.append(opr)
            values.append(output)

        # Create final graph
        outputs = [Output(v) for v in values if len(v.uses_) == 0]
        graph = Graph(inputs, outputs, oprs)

        return graph

    def _sample_op_inputs(self, op: Op, spec: TypeSpec, dtype: DataType, values: List[Value],
                          graph_inputs: List[Input]) -> List[Value]:
        # Sample number of inputs
        num_inputs = self._rng.choice(int_expr_choices(spec.in_num, 1, max_in_num + 1))

        # Sample values
        inputs = []
        for i in range(num_inputs):
            if i in op.params_:
                param = Input(self._gen_tensor_type(dtype), True)
                inputs.append(param.value_)
                graph_inputs.append(param)
            else:
                inputs.append(self._rng.choice(values))

        return inputs

    def _gen_tensor_type(self, dtype: DataType):
        rank = self._rng.integers(low=2, high=max_rank, endpoint=True)
        shape = cast(List[int],
                     self._rng.integers(low=1, high=max_dim, size=rank, endpoint=True).tolist())
        return TensorType(shape, dtype)

    def _sample_attrs(self, spec: TypeSpec) -> List[Tuple[str, ValueType]]:
        attrs = []
        for attr in spec.attrs:
            attrs.append((attr.name_, self._sample_expr(attr.expr_)))
        return attrs

    def _sample_expr(self, expr: Expr) -> ValueType:
        # Estimate scalar values
        if expr.type_.is_scalar:
            choices = expr_choices(expr, [0])
            return self._rng.choice(choices)

        # Estimate array
        if expr.kind == ExprKind.TUPLE:
            tup = cast(array.Tuple, expr)
            return tuple(self._sample_expr(f) for f in tup.fields_)
        elif expr.kind == ExprKind.LIST:
            lst = cast(array.List, expr)
            lst_len = self._sample_expr(lst.len_)
            return tuple(self._sample_expr(lst.body_) for _ in range(lst_len))
        else:
            return ()


class OpCoverage:
    def __init__(self):
        self._ops: Set[Op] = set()

    def count(self, graph: Graph):
        for opr in graph.oprs_:
            self._ops.add(opr.op_)

    def get(self):
        return len(self._ops)


args = Namespace()


def _parse_args():
    global args
    p = ArgumentParser()
    p.add_argument('-n', '--number', type=int, help='Number of graphs.')
    p.add_argument('-s', '--seed', type=int, default=42, help='Random seed of graph generator.')
    args = p.parse_args()


def main():
    # Initialization
    rng = Generator(PCG64(seed=args.seed))
    ops = [OpRegistry.get(name) for name in common_ops]
    gen = PureRandomGenerator(ops, rng)

    # Generation loop
    progress = tqdm(range(args.number), file=stdout)
    num_valid = 0
    op_cov = OpCoverage()

    def update_stat():
        progress.set_postfix_str(
            'valid={}, op_cov={:.3f}'.format(num_valid, op_cov.get() / len(ops)))

    update_stat()
    for _ in progress:
        # Generate graph
        graph = gen.generate()
        code = print_relay(graph, extra_types=False)

        # Check type correctness
        try:
            parser.parse(code)
        except TVMError:
            continue

        num_valid += 1
        op_cov.count(graph)
        update_stat()


if __name__ == '__main__':
    _parse_args()
    main()
