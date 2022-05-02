import os
import re
from argparse import Namespace, ArgumentParser

from typefuzz.debug import ErrorKind, CompileReducer, RunReducer

options = Namespace()


def parse_args():
    global options
    p = ArgumentParser()
    p.add_argument('-d', '--directory', type=str, help='Directory for storing error cases.')
    options = p.parse_args()


def main():
    level_matcher = re.compile('opt_level=(\\d)')
    for case_id in sorted(os.listdir(options.directory), key=lambda s: int(s)):
        case_path = os.path.join(options.directory, case_id)
        with open(os.path.join(case_path, 'error.txt'), 'r') as f:
            opt_str = f.readline()
            err = f.read()
        opt_level = int(next(level_matcher.finditer(opt_str)).groups()[0])
        with open(os.path.join(case_path, 'code.txt'), 'r') as f:
            code = f.read()
        for kind, reduce_cls in zip(
                [ErrorKind.COMPILE, ErrorKind.RUN],
                [CompileReducer, RunReducer]
        ):
            if not os.path.exists(os.path.join(case_path, kind.name)):
                continue
            print(f'Reducing case {case_id}:')
            reducer = reduce_cls(code, err, opt_level)
            reduced_code = reducer.reduce()
            with open(os.path.join(case_path, 'code-reduced.txt'), 'w') as f:
                f.write(reduced_code)


if __name__ == '__main__':
    parse_args()
    main()
