import os.path
from argparse import Namespace, ArgumentParser
from sys import stdout
from time import strftime

from numpy import allclose
from numpy.random import Generator, PCG64
from tqdm import tqdm
from tvm import IRModule
from tvm import relay, transform, cpu
from tvm.contrib.graph_executor import GraphModule

from typefuzz.debug import RelayRunner
from typefuzz.graph import GraphGenerator, print_relay
from typefuzz.spec import OpRegistry

options = Namespace()


def parse_args():
    global options
    p = ArgumentParser()
    p.add_argument('-s', '--seed', type=int, default=42, help='Random seed of graph generator.')
    options = p.parse_args()


def _gen_tensor(var, rng: Generator):
    var_ty = var.checked_type
    return rng.standard_normal(
        size=[int(d) for d in var_ty.shape], dtype='float64'
    ).astype(var_ty.dtype)


def build_run_relay(mod: IRModule, rng: Generator):
    # Generate input parameters
    main_fn = mod['main']
    inputs = {main_fn.params[0].name_hint: _gen_tensor(main_fn.params[0], rng)}
    params = {var.name_hint: _gen_tensor(var, rng) for var in main_fn.params[1:]}

    # Build and run unoptimized module as reference
    dev = cpu()
    with transform.PassContext(opt_level=0):
        lib = relay.build(mod, target='llvm', params=params)
    graph_exec = GraphModule(lib['default'](dev))
    graph_exec.run(**inputs)
    ref_outputs = [graph_exec.get_output(i).numpy() for i in range(graph_exec.get_num_outputs())]

    # Build and run modules with different levels of optimization
    for opt_level in range(1, 4):
        with transform.PassContext(opt_level=opt_level, disabled_pass=['AlterOpLayout']):
            lib = relay.build(mod, target='llvm', params=params)
        graph_exec = GraphModule(lib['default'](dev))
        graph_exec.run(**inputs)
        outputs = [graph_exec.get_output(i).numpy() for i in range(graph_exec.get_num_outputs())]
        for i, (o, ro) in enumerate(zip(outputs, ref_outputs)):
            if not allclose(o, ro, rtol=1e-2, atol=1e-3, equal_nan=True):
                raise ValueError(
                    f'Difference detected in output tensor {i} at optimization level {opt_level}.'
                )


def main():
    # Initialization
    rng = Generator(PCG64(seed=options.seed))
    gen = GraphGenerator(OpRegistry.ops(), rng)
    path = strftime('out/run-%Y%m%d-%H%M%S')
    if not os.path.exists(path):
        os.mkdir(path)

    # Generation loop
    progress = tqdm(file=stdout)
    while True:
        # Generate graph
        graph = gen.generate()
        relay_src = print_relay(graph)

        # Test TVM with Relay source
        RelayRunner(rng, os.path.join(path, str(progress.n))).run(relay_src)
        progress.update()


if __name__ == '__main__':
    parse_args()
    main()
