from argparse import ArgumentParser, Namespace
from sys import stdout

from numpy.random import Generator, PCG64
from onnx.shape_inference import InferenceError
from tqdm import tqdm
from tvm import TVMError

from nnsmith.materialize import Model
from nnsmith.narrow_spec import auto_opset
from nnsmith.relay_gen import nnsmith_gen_relay

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
    opset = auto_opset(Model.init('onnx'))

    # Generation loop
    progress = tqdm(range(args.number), file=stdout)
    native_invalid, relay_invalid = 0, 0

    def display_count():
        progress.set_postfix_str(f'{native_invalid}, {relay_invalid}')

    for _ in progress:
        # Generate Keras model
        try:
            nnsmith_gen_relay(opset, 32, rng)
        except TVMError:
            relay_invalid += 1
        except InferenceError as err:
            print(err)
            relay_invalid += 1
        except RuntimeError as err:
            print(err)
            pass
        display_count()

    print('Original pass rate: {:.3f}'.format(1 - native_invalid / args.number))
    print('Failure rate of conversion: {:.3f}'.format(relay_invalid / args.number))
    print('Relay pass rate: {:.3f}'.format(1 - (native_invalid + relay_invalid) / args.number))


if __name__ == '__main__':
    _parse_args()
    main()
