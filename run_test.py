import os
from argparse import Namespace, ArgumentParser
from subprocess import run, TimeoutExpired, CalledProcessError
from sys import stdout
from time import strftime

from numpy.random import Generator, PCG64
from tqdm import tqdm

from typefuzz.debug import ModuleRunner
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
        code = print_relay(graph)

        # Write code to case directory
        case_id = str(progress.n)
        case_path = os.path.join(path, case_id)
        os.mkdir(case_path)
        with open(os.path.join(case_path, 'code.txt'), 'w') as f:
            f.write(code)

        # Run subprocess
        cmd = ['python3', '_test_ps.py', f'-d={case_path}', f'-s={rng.integers(65536)}']
        try:
            run(cmd, check=True, timeout=60, stderr=open(os.devnull, 'w'))
        except CalledProcessError:
            print(f'Error detected in case {case_id}.')
        except TimeoutExpired:
            print(f'Case {case_id} timed out.')
        else:
            os.remove(os.path.join(case_path, 'code.txt'))
            os.rmdir(case_path)

        progress.update()


if __name__ == '__main__':
    parse_args()
    main()
