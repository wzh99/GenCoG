# [Bug] `tanh` outputs `1.` for `nan` input at optimization level 4

### Expected behavior

The output of `tanh` for `nan` input at optimization level 4 should be consistent with lower
optimization levels.

### Actual behavior

`tanh` outputs `1.` at optimization level 4, and `nan` at lower levels.

```
opt_level=0 [nan]
opt_level=1 [nan]
opt_level=2 [nan]
opt_level=3 [nan]
opt_level=4 [1.]
```

### Environment

macOS 12.4. Compiled using Clang 13.1.6 with LLVM support. TVM
commit [`8341e33`](https://github.com/apache/tvm/commit/8341e33d05868b7bb8496c913679b7951836f3b9).

### Steps to reproduce

```python
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
```

### Possible cause

[tvm/src/relay/transforms/fast_math.cc#L48-L49](https://github.com/apache/tvm/blob/8341e33d05868b7bb8496c913679b7951836f3b9/src/relay/transforms/fast_math.cc#L48-L49)

`tanh` is replaced by `fast_tanh` in `FastMath` pass at optimization level 4.

[tvm/include/tvm/topi/elemwise.h#L84](https://github.com/apache/tvm/blob/8341e33d05868b7bb8496c913679b7951836f3b9/include/tvm/topi/elemwise.h#L84)

If the data type of `fast_tanh` is `float32`, TOPI uses a fast float implementation of `tanh`. In
this line, the input is clamped to [-9, 9]. `nan` may be incorrectly "clamped" to 9 here. Then the
output of this implementation will be 1. 
