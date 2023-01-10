from argparse import ArgumentParser, Namespace
from sys import stdout

from numpy.random import Generator, PCG64
from tqdm import tqdm
from tvm import TVMError

from nnsmith.relay_gen import nnsmith_gen_relay, common_opset

args = Namespace()


def _parse_args():
    global args
    p = ArgumentParser()
    p.add_argument('-n', '--number', type=int, help='Number of graphs')
    p.add_argument('-s', '--seed', type=int, default=42, help='Random seed of graph generator.')
    args = p.parse_args()


def main():
    # Initialization
    rng = Generator(PCG64(seed=args.seed))

    # Generation loop
    progress = tqdm(range(args.number), file=stdout)
    native_invalid, relay_invalid = 0, 0

    def display_count():
        progress.set_postfix_str(f'{native_invalid}, {relay_invalid}')

    progress.set_postfix_str(str(native_invalid))
    for _ in progress:
        # Generate Keras model
        try:
            nnsmith_gen_relay(common_opset, 32, rng)
        except TVMError as err:
            print(err)
            relay_invalid += 1
            display_count()
        except Exception as err:
            print(err)
            native_invalid += 1
            display_count()


if __name__ == '__main__':
    _parse_args()
    main()
