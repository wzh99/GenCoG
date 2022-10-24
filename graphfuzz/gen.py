from functools import reduce
from typing import Dict, List

from numpy.random import Generator

from gencog import Const
from gencog.config import params
from gencog.expr.ty import ValueType
from gencog.graph import GraphGenerator, Value, Operation, Input, Output, Graph
from gencog.graph.lookup import ValueLookup
from gencog.solve import TypeSolver, TensorType
from gencog.solve.store import ValueStore
from gencog.spec import OpRegistry
from .dag import rn_model, ws_model
from .op import seq_ops, binary_ops

max_opr_num: int = params['graph.max_opr_num']


class GraphFuzzGenerator(GraphGenerator):
    def __init__(self, graph_model: str, rng: Generator):
        super().__init__(map(OpRegistry.get, seq_ops), rng)
        self._graph_model = graph_model
        self._bin_ops = list(map(OpRegistry.get, binary_ops))

    def generate(self):
        # Generate graph model
        if self._graph_model == 'rn':
            nodes = rn_model(max_opr_num, 0.5, self._rng)
        elif self._graph_model == 'ws':
            nodes = ws_model(max_opr_num, 2, 0.5, self._rng)
        else:
            raise ValueError(f'Unknown graph model {self._graph_model}')

        # Initialization
        inputs = []
        oprs = []
        node_value_map: Dict[int, Value] = {}

        # Generate vertices according to graph model
        for node in nodes:
            # Generate input vertex
            if node.is_input:
                init_in = self._gen_input()
                inputs.append(init_in)
                node_value_map[node.id] = init_in.value_
                continue

            # Generate operation vertex according to node type
            args = list(map(lambda node: node_value_map[node.id], node.inbound_nodes))
            if not node.is_merging:
                opr = self._gen_seq_op(args[0], inputs)
            elif len(node.inbound_nodes) == 2:
                opr = self._gen_binary_op(args[0], args[1], oprs, inputs)
            else:
                opr = self._gen_concat(args, oprs, inputs)
            oprs.append(opr)
            node_value_map[node.id] = opr.outputs_[0]

        # Create final graph
        outputs = [Output(v) for v in node_value_map.values() if len(v.uses_) == 0]
        graph = Graph(inputs, outputs, oprs)

        return graph

    def _gen_seq_op(self, data: Value, graph_inputs: List[Input]) -> Operation:
        # Choose one operator
        op = self._rng.choice(list(self._ops.by_first_type(data.type_)))
        spec = op.spec

        # Create constraint solver
        solver = TypeSolver(spec, {0: data.type_}, self._rng)
        if 'nn.conv' in op.name_:
            self._set_known_attr(solver.store_, 'groups', 1)
        info = solver.solve()

        return self._create_opr(op, info, {0: data}, ValueLookup(), graph_inputs)

    @staticmethod
    def _set_known_attr(store: ValueStore, name: str, val: ValueType):
        for n, e in store.attrs_:
            if n == name:
                e.set_defined(Const(val))

    def _gen_binary_op(self, lhs: Value, rhs: Value, oprs: List[Operation],
                       graph_inputs: List[Input]) -> Operation:
        rhs = self._align_shape(lhs, rhs, [], oprs, graph_inputs)
        op = self._bin_ops[self._rng.integers(len(self._bin_ops))]
        return Operation(op=op, attrs=[], inputs=[lhs, rhs], outputs=[Value(lhs.type_)])

    def _gen_concat(self, args: List[Value], oprs: List[Operation],
                    graph_inputs: List[Input]) -> Operation:
        aligned_args = [self._align_shape(args[0], arg, [1], oprs, graph_inputs) for arg in args]
        newshape = [
            sum(map(lambda a: a.type_.shape_[i], aligned_args)) if i == 1 else args[0].type_.shape_[
                i] for i in range(args[0].type_.rank)]
        return Operation(
            op=OpRegistry.get('concatenate'),
            attrs=[('axis', 1)],
            inputs=aligned_args,
            outputs=[Value(TensorType(newshape, args[0].type_.dtype_))]
        )

    @staticmethod
    def _align_shape(tgt: Value, src: Value, ignored_dims: List[int], oprs: List[Operation],
                     graph_inputs: List[Input]) -> Value:
        # Align rank
        dtype = src.type_.dtype_
        src_rank, tgt_rank = src.type_.rank, tgt.type_.rank
        rank_diff = src.type_.rank - tgt.type_.rank
        if rank_diff < 0:
            # Increase rank of source value by `expand_dims`
            new_shape = src.type_.shape_ + [1] * -rank_diff
            align_opr = Operation(
                op=OpRegistry.get('expand_dims'),
                attrs=[('axis', src_rank), ('num_newaxis', -rank_diff)],
                inputs=[src],
                outputs=[Value(TensorType(new_shape, dtype))]
            )
            oprs.append(align_opr)
            src = align_opr.outputs_[0]
        elif rank_diff > 0:
            # Reshape source to reduce its rank
            new_shape = src.type_.shape_[:tgt_rank - 1] + [
                reduce(int.__mul__, src.type_.shape_[tgt_rank - 1:], 1)]
            align_opr = Operation(
                op=OpRegistry.get('reshape'),
                attrs=[('newshape', tuple(new_shape))],
                inputs=[src],
                outputs=[Value(TensorType(new_shape, dtype))]
            )
            oprs.append(align_opr)
            src = align_opr.outputs_[0]

        # Try padding
        pad_size = list(map(lambda p: 0 if p[0] in ignored_dims else max(0, p[1][1] - p[1][0]),
                            enumerate(zip(src.type_.shape_, tgt.type_.shape_))))
        if sum(pad_size) != 0:
            new_shape = list(map(lambda p: p[0] + p[1], zip(src.type_.shape_, pad_size)))
            pad_value_input = Input(TensorType([], dtype), True)
            graph_inputs.append(pad_value_input)
            align_opr = Operation(
                op=OpRegistry.get('nn.pad'),
                attrs=[
                    ('pad_width', tuple(map(lambda s: [0, s], pad_size))),
                    ('pad_mode', 'constant')
                ],
                inputs=[src, pad_value_input.value_],
                outputs=[Value(TensorType(new_shape, dtype))]
            )
            oprs.append(align_opr)
            src = align_opr.outputs_[0]

        # Try slice
        slice_size = list(map(lambda p: 0 if p[0] in ignored_dims else max(0, p[1][0] - p[1][1]),
                              enumerate(zip(src.type_.shape_, tgt.type_.shape_))))
        if sum(slice_size) != 0:
            new_shape = list(map(lambda p: p[0] - p[1], zip(src.type_.shape_, slice_size)))
            align_opr = Operation(
                op=OpRegistry.get('strided_slice'),
                attrs=[
                    ('axes', tuple(range(tgt_rank))), ('begin', (0,) * tgt_rank),
                    ('end', tuple(new_shape)), ('strides', (1,) * tgt_rank)
                ],
                inputs=[src],
                outputs=[Value(TensorType(new_shape, dtype))]
            )
            oprs.append(align_opr)
            src = align_opr.outputs_[0]

        return src
