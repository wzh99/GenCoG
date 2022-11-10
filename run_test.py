import os
from argparse import Namespace, ArgumentParser
from subprocess import run, TimeoutExpired, CalledProcessError
from sys import stdout
from time import strftime

from numpy.random import Generator, PCG64
from tqdm import tqdm

from gencog.graph import GraphGenerator, print_relay
from gencog.spec import OpRegistry

args = Namespace()


def parse_args():
    global args
    p = ArgumentParser()
    p.add_argument('-r', '--root', type=str, help='Root directory of TVM source code.')
    p.add_argument('-s', '--seed', type=int, default=42, help='Random seed of graph generator.')
    p.add_argument('-o', '--output', type=str, default='out', help='Output directory.')
    args = p.parse_args()


def main():
    # Initialization
    rng = Generator(PCG64(seed=args.seed))
    gen = GraphGenerator(OpRegistry.ops(), rng)
    path = os.path.join(args.output, strftime('run-%Y%m%d-%H%M%S'))
    env = os.environ.copy()
    env['PYTHONPATH'] = os.path.join(args.root, 'python')
    if not os.path.exists(path):
        os.mkdir(path)

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
        cmd = ['python3', '_run_ps.py', f'-d={case_path}', '-e', f'-s={rng.integers(2 ** 63)}']
        keep_dir = False
        try:
            run(cmd, env=env, check=True, timeout=60, stderr=open(os.devnull, 'w'))
        except CalledProcessError:
            print(f'Error detected in case {case_id}.')
            keep_dir = True
        except TimeoutExpired:
            print(f'Case {case_id} timed out.')
        if not keep_dir:
            os.remove(os.path.join(case_path, 'code.txt'))
            os.rmdir(case_path)
        progress.update()


if __name__ == '__main__':
    parse_args()
    main()
