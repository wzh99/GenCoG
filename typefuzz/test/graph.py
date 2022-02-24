from argparse import Namespace, ArgumentParser
from sys import stdout

from numpy.random import Generator, PCG64
from tqdm import trange
from tvm import parser

from typefuzz.graph.gen import GraphGenerator
from typefuzz.graph.relay import print_relay
from typefuzz.graph.viz import visualize
from typefuzz.spec import OpRegistry, TypeSpec

options = Namespace()


def parse_args():
    global options
    p = ArgumentParser()
    p.add_argument('-n', '--number', type=int, help='Number of graphs to generate.')
    p.add_argument('-seed', type=int, default=42, help='Random seed of graph generator.')
    p.add_argument('-view', action='store_true', help='Open graph visualization after rendering.')
    options = p.parse_args()


if __name__ == '__main__':
    parse_args()
    TypeSpec.for_graph = True
    rng = Generator(PCG64(seed=options.seed))
    gen = GraphGenerator(OpRegistry.ops(), rng)
    for idx in trange(options.number, file=stdout):
        graph = gen.generate()
        src = print_relay(graph)
        parser.parse(src)
        visualize(graph, f'graph_{idx}', 'out', view=options.view)
