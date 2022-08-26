from argparse import ArgumentParser, Namespace
from sys import stdout

import numpy as np
from tqdm import tqdm
from tvm import relay, TVMError

from muffin.model_generator import ModelGenerator
from tvm_frontend import from_keras

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
    keras_invalid, relay_invalid = 0, 0

    def display_count():
        progress.set_postfix_str(f'{keras_invalid}, {relay_invalid}')

    progress.set_postfix_str(str(keras_invalid))
    for _ in progress:
        # Generate Keras model
        try:
            model = model_gen.generate(args.mode)
        except ValueError as err:
            print(err)
            keras_invalid += 1
            display_count()
            continue

        # Convert to Relay
        batch_size = np.random.randint(1, 5)
        input_shapes = {inp.name: (batch_size,) + tuple(inp.shape.as_list()[1:])
                        for inp in model.inputs}
        mod, params = from_keras(model, shape=input_shapes)

        # Check type correctness
        try:
            relay.transform.InferType()(mod)
        except TVMError:
            relay_invalid += 1
            display_count()
            continue


if __name__ == '__main__':
    _parse_args()
    main()
