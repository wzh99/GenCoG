# Confirmed: https://github.com/apache/tvm/issues/11696

import numpy as np
from tvm import relay, transform, cpu, IRModule
from tvm.contrib.graph_executor import GraphModule

x = relay.var('x', shape=(1,), dtype='float32')
y = relay.tanh(x)
mod = IRModule.from_expr(y)
for level in range(5):
    with transform.PassContext(opt_level=level):
        lib = relay.build(mod, target='llvm')
        gmod = GraphModule(lib['default'](cpu()))
        gmod.run(x=np.array([np.nan]))
        print(f'opt_level={level}', gmod.get_output(0))
