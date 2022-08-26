import time
from argparse import Namespace, ArgumentParser
from sys import stdout

import numpy as np
from numpy.random import Generator, PCG64
from tqdm import tqdm
from tvm import parser, TVMError

from gencog.config import muffin_ops
from gencog.graph import GraphGenerator, print_relay
from gencog.metric.div import EdgeDiversity, VertexDiversity
from gencog.spec import OpRegistry

args = Namespace()


def _parse_args():
    global args
    p = ArgumentParser()
    p.add_argument('--opset', type=str, choices=['all', 'muffin'],
                   help='Operator set for generating graphs.')
    p.add_argument('-l', '--limit', type=int, help='Limit on total number of operations.')
    p.add_argument('-s', '--seed', type=int, default=42, help='Random seed of graph generator.')
    args = p.parse_args()


def main():
    # Initialization
    opr_limit = args.limit
    if args.opset == 'muffin':
        ops = [OpRegistry.get(name) for name in muffin_ops]
    else:
        ops = list(OpRegistry.ops())
    rng = Generator(PCG64(seed=args.seed))
    gen = GraphGenerator(ops, rng)
    vert_div = VertexDiversity(ops)
    edge_div = EdgeDiversity(ops)

    # Generation loop
    opr_count = 0
    progress = tqdm(total=opr_limit, file=stdout)
    div_record = []
    record_file = time.strftime(f'out/gencog-{args.opset}-%Y%m%d-%H%M%S.txt')
    loop_idx = 0
    while True:
        # Generate graph
        graph = gen.generate()
        code = print_relay(graph)

        # Check type correctness
        try:
            parser.parse(code)
        except TVMError:
            continue

        # Evaluate diversity
        vert_div.evaluate(graph)
        edge_div.evaluate(graph)

        # Count operations
        opr_num = len(graph.oprs_)
        opr_count += opr_num
        progress.update(n=opr_num)

        # Write record to file
        div_record.append([opr_count, vert_div.result, edge_div.result])
        if loop_idx % 10 == 0:
            # noinspection PyTypeChecker
            np.savetxt(record_file, np.array(div_record), fmt='%.4f')

        # Stop if operation limit is reached
        if opr_count >= opr_limit:
            # noinspection PyTypeChecker
            np.savetxt(record_file, np.array(div_record), fmt='%.4f')
            progress.close()
            break
        loop_idx += 1

    # Output diversity
    np.set_printoptions(precision=3)
    print('Operator detail:', vert_div.op_div, sep='\n')
    print('Vertex diversity:', vert_div.result)
    print('Edge diversity:', edge_div.result)


if __name__ == '__main__':
    _parse_args()
    main()
