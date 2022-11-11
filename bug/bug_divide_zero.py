# Fixed: https://github.com/apache/tvm/pull/13348

import numpy as np
from tvm import relay, transform, cpu, IRModule
from tvm.contrib.graph_executor import GraphModule

x = relay.var('x', shape=(1,), dtype='float32')
y0 = x - x
y1 = y0 / y0
mod = IRModule.from_expr(y1)

inputs = {
    'x': np.empty((1,), dtype='float32')
}
params = {}

with transform.PassContext(opt_level=0):
    lib = relay.build(mod, params=dict(params), target='llvm')
gmod = GraphModule(lib['default'](cpu()))
gmod.run(**inputs)
ref_out = gmod.get_output(0).numpy()

with transform.PassContext(opt_level=1, disabled_pass=['AlterOpLayout']):
    mod, params = relay.optimize(mod, params=params, target='llvm')
    lib = relay.build(mod, params=dict(params), target='llvm')
gmod = GraphModule(lib['default'](cpu()))
gmod.run(**inputs)
opt_out = gmod.get_output(0).numpy()

print('Reference:', ref_out)
print('Optimized:', opt_out)
