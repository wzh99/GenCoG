# Fixed: https://github.com/apache/tvm/pull/9835

import numpy as np
import tvm
from tvm import relay, transform, IRModule, cpu
from tvm.contrib.graph_executor import GraphModule

print(tvm.__version__)

x = relay.var('x', shape=(1, 1, 4, 1), dtype='float32')
y = relay.nn.avg_pool2d(x, pool_size=(3, 2), strides=(2, 2), dilation=(2, 2), padding=(0, 2, 2, 0),
                        ceil_mode=True, count_include_pad=True)
mod = IRModule.from_expr(y)

inputs = {
    'x': np.ones((1, 1, 4, 1), 'float32')
}
params = {}

with transform.PassContext(opt_level=0):
    lib = relay.build(mod, params=params, target='llvm')
gmod = GraphModule(lib['default'](cpu()))
gmod.run(**inputs)
print(gmod.get_output(0).numpy())

'''
0.8.0
[[[[0.33333334]
   [0.16666667]]]]
   
0.9.0
[[[[0.33333334]
   [0.25      ]]]]
'''
