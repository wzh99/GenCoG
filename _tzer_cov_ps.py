import json
from argparse import ArgumentParser

import tvm
from tvm import tir, ir, transform, TVMError

from tzer.tvmpass import PassDependenceGraph


def tir_primfunc_to_mod(func: tir.PrimFunc,
                        target: tvm.target.Target = tvm.target.Target('llvm')) -> tvm.ir.IRModule:
    func = func.with_attr('target', target)
    func = func.with_attr('global_symbol', 'main')
    func = func.with_attr('tir.noalias', True)
    func = func.with_attr('from_legacy_te_schedule', True)

    return tvm.ir.IRModule({'main': func})


# Parse arguments
parser = ArgumentParser()
parser.add_argument('-f', '--func', type=str)
parser.add_argument('-p', '--passes', type=str)
args = parser.parse_args()

# Create dependence graph
pass_graph = PassDependenceGraph()

# Load TIR function and passes
with open(args.func) as f:
    func_dict = json.load(f)
func = ir.load_json(json.dumps(func_dict))
with open(args.passes) as f:
    pass_info_list = json.load(f)
passes = [pass_graph.get_concrete_pass(info) for info in pass_info_list]

# Optimize and compile TIR function
try:
    # noinspection PyTypeChecker
    mod = tir_primfunc_to_mod(func)
    with transform.PassContext(opt_level=4):
        for single_pass in passes:
            try:
                mod = single_pass(mod)
            except TVMError:
                pass
        opt_mod = tvm.build(mod)
except TVMError:
    pass
