import os
from argparse import Namespace, ArgumentParser
from subprocess import run, TimeoutExpired, CalledProcessError
from sys import stdout
from time import strftime
from typing import List

import numpy as np
from numpy.random import Generator, PCG64
from tqdm import tqdm

from gencog.graph import GraphGenerator, print_relay
from gencog.spec import OpRegistry

args = Namespace()


def parse_args():
    global args
    p = ArgumentParser()
    p.add_argument('-1', '--first', type=str, help='Root directory of first TVM.')
    p.add_argument('-2', '--second', type=str, help='Root directory of second TVM.')
    p.add_argument('-s', '--seed', type=int, default=42, help='Random seed of graph generator.')
    p.add_argument('-o', '--output', type=str, default='out', help='Output directory.')
    args = p.parse_args()


def load_all_results(case_dir: str, del_after_load: bool = False) -> List[List[np.ndarray]]:
    all_results = []
    for level in range(5):
        path = os.path.join(case_dir, f'O{level}.npz')
        with np.load(path) as f:
            all_results.append([f[f'arr_{i}'] for i in range(len(f))])
        if del_after_load:
            os.remove(path)
    return all_results


def main():
    # Initialization
    rng = Generator(PCG64(seed=args.seed))
    gen = GraphGenerator(OpRegistry.ops(), rng)
    path = os.path.join(args.output, strftime('run-%Y%m%d-%H%M%S'))
    env = os.environ.copy()
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

        # Run first TVM in subprocess
        keep_dir = False
        can_diff = True
        env['PYTHONPATH'] = os.path.join(args.first, 'python')
        cmd = ['python3', '_run_ps.py', f'-d={case_path}', '-r', f'-s={rng.integers(2 ** 63)}']
        try:
            run(cmd, env=env, check=True, timeout=60, stderr=open(os.devnull, 'w'))
        except CalledProcessError:
            print(f'Error detected in case {case_id}.')
            can_diff = False
        except TimeoutExpired:
            print(f'Case {case_id} timed out.')
            can_diff = False
        first_results = []
        if can_diff:
            first_results = load_all_results(case_path, del_after_load=True)

        # Run second TVM
        if can_diff:
            env['PYTHONPATH'] = os.path.join(args.second, 'python')
            try:
                run(cmd, env=env, check=True, timeout=60, stderr=open(os.devnull, 'w'))
            except CalledProcessError:
                print(f'Error detected in case {case_id}.')
                can_diff = False
            except TimeoutExpired:
                print(f'Case {case_id} timed out.')
                can_diff = False
        second_results = []
        if can_diff:
            second_results = load_all_results(case_path, del_after_load=True)

        # Compare results
        if can_diff:
            for level, (first_res, second_res) in enumerate(zip(first_results, second_results)):
                for index, (first, second) in enumerate(zip(first_res, second_res)):
                    if not np.allclose(first, second, rtol=1e-3, atol=1e-2, equal_nan=True):
                        print(f'Difference of output {index} at optimization level {level}.')
                        keep_dir = True

        # Delete case directory
        if not keep_dir:
            os.remove(os.path.join(case_path, 'code.txt'))
            os.rmdir(case_path)
        progress.update()


if __name__ == '__main__':
    parse_args()
    main()
