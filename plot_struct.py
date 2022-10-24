from argparse import Namespace, ArgumentParser
from sys import stdout

from graphviz import Digraph
from numpy.random import Generator, PCG64
from tqdm import trange

from gencog.config import params, common_ops
from gencog.graph import GraphGenerator, GraphVisitor, Graph, Input, Output, Operation
from gencog.spec import OpRegistry
from gencog.util import NameGenerator

args = Namespace()


def parse_args():
    global args
    p = ArgumentParser()
    p.add_argument('-n', '--number', type=int, help='Number of sample graphs to generate.')
    p.add_argument('-s', '--seed', type=int, default=42, help='Random seed of graph generator.')
    args = p.parse_args()


class StructurePlotter(GraphVisitor[str]):
    line_width = 8.0

    def __init__(self, name: str, directory: str):
        super().__init__()
        self._viz = Digraph(
            name=name,
            directory=directory,
            format='pdf',
            graph_attr={
                # 'nodesep': '0.05',
            },
            node_attr={
                'shape': 'circle',
                'width': '1.0',
                'penwidth': str(self.line_width),
            },
            edge_attr={
                'penwidth': str(self.line_width),
            }
        )
        self._opr_gen = NameGenerator('opr')

    def plot(self, graph: Graph):
        for out in graph.outputs_:
            self.visit(out)
        self._viz.render()

    def visit_input(self, i: Input):
        return ''

    def visit_output(self, o: Output):
        self.visit(o.value_.def_)
        return ''

    def visit_operation(self, opr: Operation):
        name = self._opr_gen.generate()
        self._viz.node(name, label='')
        for v in opr.inputs_:
            pred = v.def_
            if not isinstance(pred, Operation):
                continue
            pred_name = self.visit(pred)
            self._viz.edge(pred_name, name)
        return name


def main():
    # Initialization
    rng = Generator(PCG64(seed=args.seed))
    ops = [OpRegistry.get(name) for name in common_ops]
    gen = GraphGenerator(ops, rng)
    vert_num = params['graph.max_opr_num']
    use_penal = params['graph.use_penal']
    out_dir = f'out/struct-{vert_num}-{use_penal}'

    # Generation loop
    for idx in trange(args.number, file=stdout):
        # Generate graph
        graph = gen.generate()

        # Plot graph
        StructurePlotter(f'struct-{vert_num}-{use_penal}-{idx}', out_dir).plot(graph)


if __name__ == '__main__':
    parse_args()
    main()
