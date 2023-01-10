import time
from argparse import ArgumentParser, Namespace
from sys import stdout

import numpy as np
from numpy.random import Generator, PCG64
from tqdm import tqdm

from gencog.config import common_ops
from gencog.graph.relay import build_graph
from gencog.metric.div import VertexDiversity, EdgeDiversity
from gencog.spec import OpRegistry
from nnsmith.relay_gen import nnsmith_gen_relay, common_opset

args = Namespace()


def _parse_args():
    global args
    p = ArgumentParser()
    p.add_argument('-l', '--limit', type=int, help='Limit on total number of operations.')
    p.add_argument('-s', '--seed', type=int, default=42, help='Random seed of graph generator.')
    args = p.parse_args()


def main():
    # Initialization
    rng = Generator(PCG64(seed=args.seed))
    ops = [OpRegistry.get(name) for name in common_ops]
    vert_div = VertexDiversity(ops)
    edge_div = EdgeDiversity(ops)

    # Generation loop
    opr_count = 0
    progress = tqdm(total=args.limit, file=stdout)
    div_record = []
    record_file = time.strftime(f'out/nnsmith-%Y%m%d-%H%M%S.txt')
    loop_idx = 0
    while True:
        # Try to generate graph
        try:
            mod, params = nnsmith_gen_relay(common_opset, 32, rng)
        except Exception:
            continue
        graph = build_graph(mod, params)

        # Evaluate diversity
        vert_div.evaluate(graph)
        edge_div.evaluate(graph)

        # Count operations
        opr_num = len(graph.oprs_)
        opr_count += opr_num
        progress.update(n=opr_num)

        # Write record to file
        vd, ed = vert_div.result, edge_div.result
        div_record.append([opr_count, vd, ed])
        progress.set_postfix_str('vert={:.4f}, edge={:.4f}'.format(vd, ed))
        if loop_idx % 5 == 0:
            # noinspection PyTypeChecker
            np.savetxt(record_file, np.array(div_record), fmt='%.4f')

        # Stop if operation limit is reached
        if opr_count >= args.limit:
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
