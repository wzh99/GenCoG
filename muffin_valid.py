from argparse import ArgumentParser, Namespace
from sys import stdout

from tqdm import tqdm

from muffin.model_generator import ModelGenerator

args = Namespace()


def _parse_args():
    global args
    p = ArgumentParser()
    p.add_argument('-n', '--number', type=int, help='Number of graphs')
    p.add_argument('-m', '--mode', type=str, choices=['seq', 'merge', 'dag', 'template'],
                   help='Generation mode.')
    args = p.parse_args()


def main():
    # Initialization
    model_gen = ModelGenerator()

    # Generation loop
    progress = tqdm(range(args.number), file=stdout)
    num_invalid = 0
    progress.set_postfix_str(str(num_invalid))
    for _ in progress:
        # Generate Keras model
        try:
            model_gen.generate(args.mode)
        except ValueError as err:
            print(err)
            num_invalid += 1
            progress.set_postfix_str(str(num_invalid))


if __name__ == '__main__':
    _parse_args()
    main()
