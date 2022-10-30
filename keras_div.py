import time
from argparse import ArgumentParser, Namespace
from sys import stdout

import numpy as np
from tqdm import tqdm
from tvm import relay, TVMError

from gencog.config import common_ops
from gencog.graph.relay import build_graph
from gencog.metric.div import VertexDiversity, EdgeDiversity
from gencog.spec import OpRegistry
from lemon.gen import LemonGenerator
from muffin.model_generator import MuffinGenerator
from tvm_frontend import from_keras

args = Namespace()


def _parse_args():
    global args
    p = ArgumentParser()
    p.add_argument('-g', '--generator', type=str, choices=['lemon', 'muffin'],
                   help='Method for graph generation.')
    p.add_argument('-l', '--limit', type=int, help='Limit on total number of operations.')
    p.add_argument('-m', '--model', type=str, choices=['seq', 'merge', 'dag', 'template'],
                   help='Graph model to apply, only valid for Muffin.')
    args = p.parse_args()


def main():
    # Initialization
    opr_limit = args.limit
    if args.generator == 'lemon':
        model_gen = LemonGenerator()
    else:
        model_gen = MuffinGenerator(args.model)
    ops = [OpRegistry.get(name) for name in common_ops]
    vert_div = VertexDiversity(ops)
    edge_div = EdgeDiversity(ops)

    # Generation loop
    opr_count = 0
    progress = tqdm(total=opr_limit, file=stdout)
    div_record = []
    if args.generator == 'lemon':
        record_file = time.strftime(f'out/lemon-%Y%m%d-%H%M%S.txt')
    else:
        record_file = time.strftime(f'out/muffin-{args.mode}-%Y%m%d-%H%M%S.txt')
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

        # Evaluate diversity
        vert_div.evaluate(graph)
        edge_div.evaluate(graph)

        # Count operations
        opr_num = len(graph.oprs_)
        opr_count += opr_num
        progress.update(n=opr_num)

        # Write record to file
        div_record.append([opr_count, vert_div.result, edge_div.result])
        # noinspection PyTypeChecker
        np.savetxt(record_file, np.array(div_record), fmt='%.4f')

        # Stop if operation limit is reached
        if opr_count >= opr_limit:
            progress.close()
            break

    # Output diversity
    np.set_printoptions(precision=3)
    print('Operator detail:', vert_div.op_div, sep='\n')
    print('Vertex diversity:', vert_div.result)
    print('Edge diversity:', edge_div.result)


if __name__ == '__main__':
    _parse_args()
    main()
