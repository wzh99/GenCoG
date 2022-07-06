import time
from argparse import ArgumentParser, Namespace
from sys import stdout

import numpy as np
from tqdm import tqdm
from tvm.parser import parse

from muffin.model_generator import ModelGenerator
from tvm_util.frontend import from_keras
from typefuzz.config import muffin_ops
from typefuzz.graph.relay import build_graph
from typefuzz.metric.div import VertexDiversity, EdgeDiversity
from typefuzz.spec import OpRegistry
from typefuzz.util import run_process

_gen_modes = ['seq', 'merge', 'dag', 'template']

args = Namespace()


def _parse_args():
    global args
    p = ArgumentParser()
    p.add_argument('-l', '--limit', type=int, help='Limit on total number of operations.')
    p.add_argument('-m', '--mode', type=str, choices=_gen_modes, help='Generation mode.')
    args = p.parse_args()


def _check_relay(src: str):
    mod = parse(src)
    return {'src': mod.astext()}


def main():
    # Initialization
    opr_limit = args.limit
    model_gen = ModelGenerator()
    ops = [OpRegistry.get(name) for name in muffin_ops]
    vert_div = VertexDiversity(ops)
    edge_div = EdgeDiversity(ops)

    # Generation loop
    opr_count = 0
    progress = tqdm(total=opr_limit, file=stdout)
    div_record = []
    record_file = time.strftime("out/muffin-%Y%m%d-%H%M%S.txt", time.localtime())
    while True:
        # Generate Keras model
        mode = args.mode
        try:
            model = model_gen.generate(mode)
        except ValueError:
            # print('Generation failed:', err)
            continue

        # Convert to Relay
        batch_size = np.random.randint(1, 5)
        input_shapes = {inp.name: (batch_size,) + tuple(inp.shape.as_list()[1:])
                        for inp in model.inputs}
        mod, params = from_keras(model, shape=input_shapes)

        # Check type correctness
        ps_result = run_process(_check_relay, (mod.astext(),))
        if ps_result.exitcode != 0:
            continue
        mod = parse(ps_result.ret['src'])

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
