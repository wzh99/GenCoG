from argparse import Namespace, ArgumentParser
from sys import stdout

from numpy.random import Generator, PCG64
from tqdm import tqdm
from tvm import parser, TVMError

from gencog.config import common_ops
from gencog.graph import GraphGenerator, print_relay
from gencog.spec import OpRegistry
from graphfuzz.gen import GraphFuzzGenerator

args = Namespace()


def _parse_args():
    global args
    p = ArgumentParser()
    p.add_argument('-g', '--generator', type=str, choices=['gencog', 'graphfuzz'])
    p.add_argument('-n', '--number', type=int, help='Number of graphs.')
    p.add_argument('--opset', type=str, choices=['all', 'common'], default='common',
                   help='Operator set for graph generation, only valid for GenCoG.')
    p.add_argument('-m', '--model', type=str, choices=['ws', 'rn'],
                   help='Graph model to apply, only valid for GraphFuzz (Luo et al.).')
    p.add_argument('-s', '--seed', type=int, default=42, help='Random seed of graph generator.')
    args = p.parse_args()


def main():
    # Initialization
    rng = Generator(PCG64(seed=args.seed))
    if args.generator == 'gencog':
        if args.opset == 'common':
            ops = [OpRegistry.get(name) for name in common_ops]
        else:
            ops = OpRegistry.ops()
        gen = GraphGenerator(ops, rng)
    else:
        gen = GraphFuzzGenerator(args.model, rng)

    # Generation loop
    progress = tqdm(range(args.number), file=stdout)
    num_invalid = 0
    progress.set_postfix_str(str(num_invalid))
    for _ in progress:
        # Generate graph
        graph = gen.generate()
        code = print_relay(graph)

        # Check type correctness
        try:
            parser.parse(code)
        except TVMError:
            num_invalid += 1
            progress.set_postfix_str(str(num_invalid))


if __name__ == '__main__':
    _parse_args()
    main()
