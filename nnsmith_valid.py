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
    native_invalid, convert_invalid = 0, 0

    def display_count():
        progress.set_postfix_str(f'{native_invalid}, {convert_invalid}')

    for _ in progress:
        # Generate Keras model
        try:
            nnsmith_gen_relay(opset, 32, rng)
        except TVMError:
            convert_invalid += 1
        except InferenceError as err:
            print(err)
            convert_invalid += 1
        except RuntimeError as err:
            print(err)
            pass
        except Exception as err:
            print(err.__class__, err)
            native_invalid += 1
        display_count()


if __name__ == '__main__':
    _parse_args()
    main()
