# Reported: https://github.com/apache/tvm/issues/11697

import numpy as np
from tvm import relay, transform, cpu, IRModule
from tvm.contrib.graph_executor import GraphModule

x_shape = (1, 4)
b_shape = (4,)

x = relay.var('x', shape=x_shape)
b = relay.var('b', shape=b_shape)
y = relay.nn.bias_add(x, b)
y = relay.squeeze(y, axis=[])
mod = IRModule.from_expr(y)
with transform.PassContext(opt_level=1):
    lib = relay.build(mod, target='llvm')
gmod = GraphModule(lib['default'](cpu()))

x_input = np.zeros(shape=x_shape, dtype='float32')
b_input = np.zeros(shape=b_shape, dtype='float32')
gmod.run(x=x_input, b=b_input)
