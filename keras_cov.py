import json
import os
from argparse import Namespace, ArgumentParser
from subprocess import run, TimeoutExpired
from sys import stdout
from time import strftime

import numpy as np
from tqdm import tqdm
from tvm import relay, TVMError

from gencog.graph.relay import build_graph
from lemon.gen import LemonGenerator
from muffin.model_generator import MuffinGenerator
from tvm_frontend import from_keras

args = Namespace()


def parse_args():
    global args
    p = ArgumentParser()
    p.add_argument('-r', '--root', type=str, help='Root directory of TVM source code.')
    p.add_argument('-g', '--generator', type=str, choices=['lemon', 'muffin'],
                   help='Method for graph generation.')
    p.add_argument('-m', '--model', type=str, choices=['seq', 'merge', 'dag', 'template'],
                   help='Graph model to apply, only valid for Muffin.')
    p.add_argument('-l', '--limit', type=int, help='Limit on total number of vertices.')
    p.add_argument('-s', '--step', type=int,
                   help='Number of vertices between two coverage collections.')
    p.add_argument('-o', '--output', type=str, default='out', help='Output directory.')
    args = p.parse_args()


def get_line_cov(root: str, out_dir: str, delete_gcda: bool):
    cov_path = os.path.join(out_dir, 'cov.json')
    cmd = ['gcovr', '-r', root, '--json-summary-pretty', '-o', cov_path]
    if delete_gcda:
        cmd.append('-d')
    run(cmd, check=True)
    with open(cov_path, 'r') as f:
        cov = json.load(f)['line_covered']
    return cov


def main():
    # Initialization
    if args.generator == 'lemon':
        model_gen = LemonGenerator()
    else:
        model_gen = MuffinGenerator(args.model)
    if args.generator == 'lemon':
        cov_dir = os.path.join(args.output, strftime(f'cov-lemon-%Y%m%d-%H%M%S'))
    else:
        cov_dir = os.path.join(args.output, strftime(f'cov-muffin-{args.mode}-%Y%m%d-%H%M%S'))
    if not os.path.exists(cov_dir):
        os.mkdir(cov_dir)
    env = os.environ.copy()
    env['PYTHONPATH'] = os.path.join(args.root, 'python')

    # Generation loop
    progress = tqdm(total=args.limit, file=stdout)
    cov_data = []
    last_cov_num = 0

    def update_cov(delete_gcda: bool):
        progress.set_postfix_str('gcovr')
        cov = get_line_cov(args.root, cov_dir, delete_gcda)
        progress.set_postfix_str(f'cov={cov}')
        cov_data.append([progress.n, cov])
        # noinspection PyTypeChecker
        np.savetxt(os.path.join(cov_dir, 'data.txt'), np.array(cov_data), fmt='%d')

    while True:
        # Generate Keras model
        try:
            model = model_gen.generate()
        except ValueError:
            continue

        # Convert to Relay
        batch_size = np.random.randint(1, 5)
        input_shapes = {inp.name: (batch_size,) + tuple(inp.shape.as_list()[1:])
                        for inp in model.inputs}
        mod, params = from_keras(model, shape=input_shapes)

        # Check type correctness
        try:
            mod = relay.transform.InferType()(mod)
        except TVMError:
            continue

        # Convert to graph representation
        graph = build_graph(mod, params)

        # Write code to output directory
        with open(os.path.join(cov_dir, 'code.txt'), 'w') as f:
            f.write(mod.astext())

        # Run subprocess
        cmd = ['python3', '_run_ps.py', f'-d={cov_dir}']
        try:
            run(cmd, env=env, timeout=600, stderr=open(os.devnull, 'w'))
        except TimeoutExpired:
            print('Timeout')
            continue

        # Clean up
        os.remove(os.path.join(cov_dir, 'code.txt'))
        progress.update(n=len(graph.oprs_))

        # Stop if vertex limit is reached
        if progress.n >= args.limit:
            update_cov(True)
            progress.close()
            break

        # Collect coverage if enough vertices are generated
        step = progress.n - last_cov_num
        if (progress.n <= args.step and step >= args.step // 10) or step >= args.step:
            update_cov(False)
            last_cov_num = progress.n


if __name__ == '__main__':
    parse_args()
    main()
