import os.path
from argparse import Namespace, ArgumentParser
from sys import stdout
from time import strftime

from numpy.random import Generator, PCG64
from tqdm import tqdm

from typefuzz.debug import ModuleRunner, ModuleError
from typefuzz.graph import GraphGenerator, print_relay
from typefuzz.spec import OpRegistry

options = Namespace()


def parse_args():
    global options
    p = ArgumentParser()
    p.add_argument('-s', '--seed', type=int, default=42, help='Random seed of graph generator.')
    p.add_argument('-o', '--output', type=str, default='out', help='Output directory.')
    options = p.parse_args()


def main():
    # Initialization
    rng = Generator(PCG64(seed=options.seed))
    gen = GraphGenerator(OpRegistry.ops(), rng)
    path = os.path.join(options.output, strftime('run-%Y%m%d-%H%M%S'))
    if not os.path.exists(path):
        os.mkdir(path)
    runner = ModuleRunner(rng)

    # Generation loop
    progress = tqdm(file=stdout)
    while True:
        # Generate graph
        graph = gen.generate()
        relay_src = print_relay(graph)

        # Test TVM with Relay source
        try:
            runner.run(relay_src)
        except ModuleError as err:
            err.report(os.path.join(path, str(progress.n)))
        progress.update()


if __name__ == '__main__':
    parse_args()
    main()
