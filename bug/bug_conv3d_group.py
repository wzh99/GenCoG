# Fixed: https://github.com/apache/tvm/pull/12500

import numpy as np
from tvm import relay, transform, cpu, IRModule
from tvm.contrib.graph_executor import GraphModule

x = relay.var('x', shape=(2, 3, 2, 2, 3), dtype='float32')
w = relay.var('w', shape=(3, 1, 3, 1, 2), dtype='float32')
y = relay.nn.conv3d(x, w, strides=[1, 2, 2], padding=[0, 0, 1, 2, 0, 2], dilation=[1, 2, 1],
                    groups=3, channels=3, kernel_size=[3, 1, 2])
mod = IRModule.from_expr(y)

inputs = {
    'x': np.ones((2, 3, 2, 2, 3), 'float32')
}
params = {
    'w': np.ones((3, 1, 3, 1, 2), 'float32')
}

with transform.PassContext(opt_level=0):
    lib = relay.build(mod, params=dict(params), target='llvm')
gmod = GraphModule(lib['default'](cpu()))
gmod.run(**inputs)
ref_out = gmod.get_output(0).numpy()
print(ref_out)
