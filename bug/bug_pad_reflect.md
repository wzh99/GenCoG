# [Bug] Wrong padded values of `nn.pad` with `pad_mode='reflect'`

### Expected behavior

`nn.pad` in the following Relay program should output the same result as its counterpart in NumPy,
given input `[[1., 1.]]`:

```
def @main(%x: Tensor[(1, 2), float32]) {
  nn.pad(%x, 0, pad_width=[[1, 1], [1, 1]], pad_mode="reflect")
}
```

### Actual behavior

`nn.pad` in TVM outputs different result from `pad` in NumPy:

```
TVM:
 [[ 4.5864499e-41 -7.9918004e-28  4.5864499e-41 -7.9918004e-28]
 [ 1.0000000e+00  1.0000000e+00  1.0000000e+00  1.0000000e+00]
 [ 2.2953549e-40 -7.9916771e-28  2.2953549e-40 -7.9916771e-28]]
NumPy:
 [[1. 1. 1. 1.]
 [1. 1. 1. 1.]
 [1. 1. 1. 1.]]
```

Padded values in the result given by `nn.pad` in TVM seem randomized and differ between two runs.

### Environment

macOS 12.4. Compiled using Clang 13.1.6 with LLVM support. TVM
commit [`8341e33`](https://github.com/apache/tvm/commit/8341e33d05868b7bb8496c913679b7951836f3b9).

### Steps to reproduce

```python
import numpy as np
from tvm import relay, transform, cpu, IRModule
from tvm.contrib.graph_executor import GraphModule

x_shape = (1, 2)
x_input = np.ones(x_shape, dtype='float32')
y = relay.nn.pad(relay.var('x', shape=x_shape), [[1, 1], [1, 1]], pad_mode='reflect')
mod = IRModule.from_expr(y)
with transform.PassContext(opt_level=0):
    lib = relay.build(mod, target='llvm')
gmod = GraphModule(lib['default'](cpu()))
gmod.run(x=x_input)
print('TVM:\n', gmod.get_output(0))
print('NumPy:\n', np.pad(x_input, [[1, 1], [1, 1]], mode='reflect'))
```
