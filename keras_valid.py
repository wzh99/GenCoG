from argparse import ArgumentParser, Namespace
from sys import stdout

import numpy as np
from tqdm import tqdm
from tvm import relay, TVMError

from lemon.gen import LemonGenerator
from muffin.model_generator import MuffinGenerator
from tvm_frontend import from_keras

args = Namespace()


def _parse_args():
    global args
    p = ArgumentParser()
    p.add_argument('-g', '--generator', type=str, choices=['lemon', 'muffin'],
                   help='Method for graph generation.')
    p.add_argument('-n', '--number', type=int, help='Number of graphs')
    p.add_argument('-m', '--model', type=str, choices=['dag', 'template'],
                   help='Graph model to apply, only valid for Muffin.')
    args = p.parse_args()


def main():
    # Initialization
    if args.generator == 'lemon':
        model_gen = LemonGenerator()
    else:
        model_gen = MuffinGenerator(args.model)

    # Generation loop
    progress = tqdm(range(args.number), file=stdout)
    keras_invalid, relay_invalid = 0, 0

    def display_count():
        progress.set_postfix_str(f'{keras_invalid}, {relay_invalid}')

    display_count()
    for _ in progress:
        # Generate Keras model
        try:
            model = model_gen.generate()
        except ValueError as err:
            print(err)
            keras_invalid += 1
            display_count()
            continue

        # Convert to Relay
        batch_size = np.random.randint(1, 5)
        input_shapes = {inp.name: (batch_size,) + tuple(inp.shape.as_list()[1:])
                        for inp in model.inputs}
        mod, _ = from_keras(model, shape=input_shapes)

        # Check type correctness
        try:
            relay.transform.InferType()(mod)
        except TVMError:
            relay_invalid += 1
            display_count()

    print('Original pass rate: {:.3f}'.format(1 - keras_invalid / args.number))
    print('Failure rate of conversion: {:.3f}'.format(relay_invalid / args.number))
    print('Relay pass rate: {:.3f}'.format(1 - (keras_invalid + relay_invalid) / args.number))


if __name__ == '__main__':
    _parse_args()
    main()
