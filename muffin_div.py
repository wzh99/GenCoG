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
from typefuzz.spec import TypeSpec, OpRegistry
from typefuzz.util import run_process

_gen_modes = ['seq', 'merge', 'dag', 'template']

options = Namespace()


def _parse_args():
    global options
    p = ArgumentParser()
    p.add_argument('-l', '--limit', type=int, help='Limit on total number of operations.')
    p.add_argument('-m', '--mode', type=str, choices=_gen_modes + ['hybrid'],
                   help='Generation mode.')
    options = p.parse_args()


def _check_relay(src: str):
    mod = parse(src)
    return {'src': mod.astext()}


def main():
    # Initialization
    TypeSpec.for_graph = True
    opr_limit = options.limit
    model_gen = ModelGenerator()
    ops = [OpRegistry.get(name) for name in muffin_ops]
    vert_div = VertexDiversity(ops)
    edge_div = EdgeDiversity(ops)

    # Generation loop
    opr_count = 0
    progress = tqdm(total=opr_limit, file=stdout)
    div_record = []
    record_file = 'out/muffin-{}.txt'.format(time.strftime("%Y%m%d-%H%M%S", time.localtime()))
    while True:
        if options.mode == 'hybrid':
            mode = np.random.choice(_gen_modes)
        else:
            mode = options.mode

        # Generate Keras model
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
            break

    # Output diversity
    np.set_printoptions(precision=3)
    print('Operator detail:', vert_div.op_div, sep='\n')
    print('Vertex diversity:', vert_div.result)
    print('Edge diversity:', edge_div.result)


if __name__ == '__main__':
    _parse_args()
    main()

"""
[0.1   0.098 0.032 0.025 0.032 0.025 0.028 0.066 0.155 0.022 0.121 0.11
 0.031 0.03  0.201 0.113 0.091 0.047 0.004 0.012 0.003 0.124 0.029 0.007
 0.119 0.027 0.007 0.318 0.211 0.074 0.182 0.215 0.079 0.063 0.022 0.249
 0.086 1.    0.452]
Vertex diversity: 0.11820851272891202
Edge diversity: 0.3879026955950033
50020it [2:23:31,  5.81it/s]
"""
