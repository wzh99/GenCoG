# Reported: https://github.com/apache/tvm/issues/11687

import numpy as np
from tvm import relay, transform, cpu, IRModule
from tvm.contrib.graph_executor import GraphModule

x_shape = (1, 2)
y = relay.nn.pad(relay.var('x', shape=x_shape), [[1, 1], [1, 1]], pad_mode='reflect')
mod = IRModule.from_expr(y)
with transform.PassContext(opt_level=0):
    lib = relay.build(mod, target='llvm')
gmod = GraphModule(lib['default'](cpu()))
x_input = np.ones(x_shape, dtype='float32')
gmod.run(x=x_input)
print('TVM:\n', gmod.get_output(0))
print('NumPy:\n', np.pad(x_input, [[1, 1], [1, 1]], mode='reflect'))
