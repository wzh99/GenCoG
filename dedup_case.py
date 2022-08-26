import os
from argparse import Namespace, ArgumentParser

from polyleven import levenshtein

from gencog.debug.run import ErrorKind

args = Namespace()


def parse_args():
    global args
    p = ArgumentParser()
    p.add_argument('-d', '--directory', type=str, help='Directory for storing error cases.')
    args = p.parse_args()


class CaseDedup:
    def __init__(self):
        self._history = []

    def is_dup(self, err: str):
        if any(levenshtein(err, his) < 100 for his in self._history):
            return True
        else:
            self._history.append(err)
            return False


def main():
    compile_dedup = CaseDedup()
    run_dedup = CaseDedup()
    for case_id in sorted(os.listdir(args.directory), key=lambda s: int(s)):
        case_path = os.path.join(args.directory, case_id)
        with open(os.path.join(case_path, 'error.txt'), 'r') as f:
            err = f.read()
        for kind, dedup in zip(
                [ErrorKind.COMPILE, ErrorKind.RUN], [compile_dedup, run_dedup]
        ):
            if not os.path.exists(os.path.join(case_path, kind.name)):
                continue
            if dedup.is_dup(err):
                for filename in os.listdir(case_path):
                    os.remove(os.path.join(case_path, filename))
                os.rmdir(case_path)
                print(f'Case {case_id} removed.')


if __name__ == '__main__':
    parse_args()
    main()
