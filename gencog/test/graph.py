from argparse import Namespace, ArgumentParser
from sys import stdout

from numpy.random import Generator, PCG64
from tqdm import trange
from tvm import parser

from gencog.config import common_ops
from gencog.graph import GraphGenerator, print_relay, visualize
from gencog.spec import OpRegistry, TypeSpec

args = Namespace()


def parse_args():
    global args
    p = ArgumentParser()
    p.add_argument('-n', '--number', type=int, help='Number of graphs to generate.')
    p.add_argument('-s', '--seed', type=int, default=42, help='Random seed of graph generator.')
    p.add_argument('-v', '--visualize', action='store_true', help='Visualize generated graphs.')
    args = p.parse_args()


if __name__ == '__main__':
    parse_args()
    TypeSpec.for_graph = True
    rng = Generator(PCG64(seed=args.seed))
    gen = GraphGenerator((OpRegistry.get(name) for name in common_ops), rng)
    for idx in trange(args.number, file=stdout):
        graph = gen.generate()
        src = print_relay(graph)
        parser.parse(src)
        if args.visualize:
            visualize(graph, f'graph_{idx}', 'out')
