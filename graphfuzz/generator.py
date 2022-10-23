from typing import Dict, List

from numpy.random import Generator

from gencog.config import params
from gencog.graph import GraphGenerator, Value, Operation, Input, Output, Graph
from gencog.graph.lookup import ValueLookup
from gencog.solve import TypeSolver
from gencog.spec import OpRegistry
from .dag import rn_model
from .op import seq_ops

max_opr_num: int = params['graph.max_opr_num']


class GraphFuzzGenerator(GraphGenerator):
    def __init__(self, gen_mode: str, rng: Generator):
        super().__init__(map(lambda n: OpRegistry.get(n), seq_ops), rng)
        self._gen_mode = gen_mode

    def generate(self):
        # Generate graph model
        nodes = rn_model(max_opr_num)

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
                opr = self._gen_binary_op(args[0], args[1], inputs)
            else:
                opr = self._gen_concat(args, inputs)
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
        info = solver.solve()

        return self._create_opr(op, info, {0: data}, ValueLookup(), graph_inputs)

    def _gen_binary_op(self, lhs: Value, rhs: Value, graph_inputs: List[Input]) -> Operation:
        pass

    def _gen_concat(self, args: List[Value], graph_inputs: List[Input]) -> Operation:
        pass
