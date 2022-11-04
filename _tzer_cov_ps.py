import json
from argparse import ArgumentParser

import tvm
from tvm import ir, transform, TVMError

from tzer.tvmpass import PassDependenceGraph
from tzer_cov import tir_primfunc_to_mod

parser = ArgumentParser()
parser.add_argument('-f', '--func', type=str)
parser.add_argument('-p', '--passes', type=str)
args = parser.parse_args()

pass_graph = PassDependenceGraph()

with open(args.func) as f:
    func_dict = json.load(f)
func = ir.load_json(json.dumps(func_dict))
with open(args.passes) as f:
    pass_info_list = json.load(f)
passes = [pass_graph.get_concrete_pass(info) for info in pass_info_list]

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
