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
