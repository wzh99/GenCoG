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

options = Namespace()


def _parse_args():
    global options
    p = ArgumentParser()
    p.add_argument('-l', '--limit', type=int, help='Limit on total number of operations.')
    p.add_argument('-m', '--mode', type=str, choices=_gen_modes, help='Generation mode.')
    options = p.parse_args()


def _check_relay(src: str):
    mod = parse(src)
    return {'src': mod.astext()}


def main():
    # Initialization
    opr_limit = options.limit
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
        mode = options.mode
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

"""
seq
Operator detail:
[0.056 0.055 0.    0.    0.    0.    0.    0.013 0.108 0.    0.064 0.08
 0.015 0.025 0.11  0.069 0.056 0.028 0.002 0.006 0.002 0.035 0.008 0.002
 0.033 0.007 0.002 0.281 0.141 0.04  0.12  0.129 0.043 0.043 0.014 0.117
 0.058 1.    0.095]
Vertex diversity: 0.07318497543696631
Edge diversity: 0.27613412228796846


merge
Operator detail:
[0.113 0.114 0.033 0.016 0.034 0.027 0.032 0.05  0.214 0.039 0.148 0.128
 0.034 0.035 0.214 0.132 0.123 0.056 0.004 0.015 0.003 0.083 0.016 0.004
 0.075 0.015 0.004 0.359 0.251 0.088 0.167 0.259 0.083 0.082 0.031 0.301
 0.101 1.    0.325]
Vertex diversity: 0.12324792983878317
Edge diversity: 0.3938198553583169


dag
Operator detail:
[0.085 0.089 0.048 0.04  0.044 0.041 0.047 0.059 0.168 0.04  0.115 0.099
 0.026 0.029 0.175 0.099 0.113 0.043 0.003 0.009 0.002 0.074 0.012 0.003
 0.061 0.014 0.003 0.339 0.216 0.074 0.135 0.189 0.065 0.067 0.023 0.239
 0.085 1.    0.451]
Vertex diversity: 0.11339557305009637
Edge diversity: 0.388560157790927


template
Operator detail:
[0.1   0.098 0.032 0.025 0.032 0.025 0.028 0.066 0.155 0.022 0.121 0.11
 0.031 0.03  0.201 0.113 0.091 0.047 0.004 0.012 0.003 0.124 0.029 0.007
 0.119 0.027 0.007 0.318 0.211 0.074 0.182 0.215 0.079 0.063 0.022 0.249
 0.086 1.    0.452]
Vertex diversity: 0.11820851272891202
Edge diversity: 0.3879026955950033
"""
