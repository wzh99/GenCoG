import os
import sys
from argparse import Namespace, ArgumentParser
from subprocess import run, CalledProcessError, TimeoutExpired

from tqdm import tqdm

_FUNC_BY_ITER_NAME_ = 'func_by_iter_'
_PASS_BY_ITER_NAME_ = 'pass_by_iter_'

args = Namespace()


def parse_args():
    global args
    p = ArgumentParser()
    p.add_argument('--root', type=str, help='Root directory of TVM source code.')
    p.add_argument('--report-dir', type=str, help='Path to the report folder')
    args = p.parse_args()


def compile_all(dump_folder: str):
    env = os.environ.copy()
    env['PYTHONPATH'] = os.path.join(args.root, 'python')

    pbar = tqdm(file=sys.stdout)
    while True:
        it = pbar.n
        func_path = ''.join(
            [os.path.join(dump_folder, _FUNC_BY_ITER_NAME_), str(it), ".json"])
        passes_path = ''.join(
            [os.path.join(dump_folder, _PASS_BY_ITER_NAME_), str(it), ".json"])
        if not os.path.exists(func_path):  # no func left
            break

        cmd = ['python3', '_tzer_cov_ps.py', '-f', func_path, '-p', passes_path]
        try:
            run(cmd, env=env, check=True, timeout=10, stderr=open(os.devnull, 'w'))
        except CalledProcessError:
            pass
        except TimeoutExpired:
            pass

        pbar.update()


if __name__ == '__main__':
    parse_args()
    compile_all(os.path.join(args.report_dir, "json_dump"))