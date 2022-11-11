# [Bug] Inconsistent results of a Relay program at optimization level 3

### Expected behavior

The following Relay program should produce consistent results between optimization level 0 and 3:

```
#[version = "0.0.5"]
def @main(%x0: Tensor[(1, 4, 3, 2), float32] /* ty=Tensor[(1, 4, 3, 2), float32] span=from_string:4:15 */, %x1: float32 /* ty=float32 span=from_string:4:20 */, %x2: Tensor[(4), float32] /* ty=Tensor[(4), float32] span=from_string:7:31 */, %x3: Tensor[(4), float32] /* ty=Tensor[(4), float32] span=from_string:7:36 */, %x4: Tensor[(4, 1, 1, 3, 2), float32] /* ty=Tensor[(4, 1, 1, 3, 2), float32] span=from_string:13:24 */, %x5: Tensor[(4), float32] /* ty=Tensor[(4), float32] span=from_string:14:28 */, %x6: Tensor[(4), float32] /* ty=Tensor[(4), float32] span=from_string:14:33 */, %x7: Tensor[(4), float32] /* ty=Tensor[(4), float32] span=from_string:14:38 */, %x8: Tensor[(4), float32] /* ty=Tensor[(4), float32] span=from_string:14:43 */, %x9: Tensor[(4), float32] /* ty=Tensor[(4), float32] span=from_string:16:28 */, %x10: Tensor[(4), float32] /* ty=Tensor[(4), float32] span=from_string:16:33 */, %x11: Tensor[(4), float32] /* ty=Tensor[(4), float32] span=from_string:16:39 */, %x12: Tensor[(4), float32] /* ty=Tensor[(4), float32] span=from_string:16:45 */, %x13: Tensor[(4), float32] /* ty=Tensor[(4), float32] span=from_string:22:33 */, %x14: Tensor[(4), float32] /* ty=Tensor[(4), float32] span=from_string:22:39 */) {
  %0 = nn.pad(%x0, %x1, pad_width=[[0, 0], [0, 0], [2, 2], [1, 0]]) /* ty=Tensor[(1, 4, 7, 3), float32] span=from_string:5:20 */;
  %1 = expand_dims(%0, axis=3) /* ty=Tensor[(1, 4, 7, 1, 3), float32] span=from_string:19:13 */;
  %2 = nn.adaptive_avg_pool3d(%1, output_size=[1, 1, 1]) /* ty=Tensor[(1, 4, 1, 1, 1), float32] span=from_string:8:17 */;
  %3 = nn.instance_norm(%2, %x2, %x3) /* ty=Tensor[(1, 4, 1, 1, 1), float32] span=from_string:8:13 */;
  %4 = (%3, %3, %2) /* ty=(Tensor[(1, 4, 1, 1, 1), float32], Tensor[(1, 4, 1, 1, 1), float32], Tensor[(1, 4, 1, 1, 1), float32]) span=from_string:9:20 */;
  %5 = concatenate(%4, axis=1) /* ty=Tensor[(1, 12, 1, 1, 1), float32] span=from_string:10:12 */;
  %6 = cos(%5) /* ty=Tensor[(1, 12, 1, 1, 1), float32] span=from_string:11:12 */;
  %7 = sum(%6, axis=[1, 2, 3], keepdims=True) /* ty=Tensor[(1, 1, 1, 1, 1), float32] span=from_string:12:15 */;
  %8 = round(%7) /* ty=Tensor[(1, 1, 1, 1, 1), float32] span=from_string:20:23 */;
  %9 = nn.conv3d(%8, %x4, strides=[1, 2, 2], padding=[0, 1, 2, 0, 1, 0], dilation=[1, 1, 2], channels=4, kernel_size=[1, 3, 2]) /* ty=Tensor[(1, 4, 1, 1, 1), float32] span=from_string:14:23 */;
  %10 = nn.batch_norm(%9, %x5, %x6, %x7, %x8, center=False, scale=False) /* ty=(Tensor[(1, 4, 1, 1, 1), float32], Tensor[(4), float32], Tensor[(4), float32]) span=from_string:15:9 */;
  %11 = %10.0 /* ty=Tensor[(1, 4, 1, 1, 1), float32] span=from_string:16:23 */;
  %12 = nn.batch_norm(%11, %x9, %x10, %x11, %x12, center=False, scale=False) /* ty=(Tensor[(1, 4, 1, 1, 1), float32], Tensor[(4), float32], Tensor[(4), float32]) span=from_string:17:9 */;
  %13 = %12.0 /* ty=Tensor[(1, 4, 1, 1, 1), float32] span=from_string:18:13 */;
  %14 = sum(%1, axis=[2], keepdims=True) /* ty=Tensor[(1, 4, 1, 1, 3), float32] span=from_string:20:18 */;
  %15 = exp(%13) /* ty=Tensor[(1, 4, 1, 1, 1), float32] span=from_string:21:13 */;
  %16 = maximum(%14, %8) /* ty=Tensor[(1, 4, 1, 1, 3), float32] span=from_string:21:18 */;
  %17 = add(%15, %16) /* ty=Tensor[(1, 4, 1, 1, 3), float32] span=from_string:22:28 */;
  %18 = nn.instance_norm(%17, %x13, %x14) /* ty=Tensor[(1, 4, 1, 1, 3), float32] span=from_string:23:4 */;
  (%18,) /* ty=(Tensor[(1, 4, 1, 1, 3), float32],) span=from_string:4:3 */
}
```

### Actual behavior

The results are inconsistent:

```
Reference:
 [[[[[0.0242452  0.0242452  0.0242452 ]]]


  [[[0.40037617 0.40037617 0.40037617]]]


  [[[0.78425235 0.78425235 0.78425235]]]


  [[[0.75829774 0.75829774 0.75829774]]]]]
Optimized:
 [[[[[0.0242452  0.0242452  0.0242452 ]]]


  [[[0.40037617 0.40037617 0.40037617]]]


  [[[0.78425235 0.78425235 0.78425235]]]


  [[[0.19935817 0.19935817 0.19935817]]]]]
```

### Environment

Windows 11 22H2, TVM commit [f3eb239](https://github.com/apache/tvm/commit/f3eb2399897b830cd5a4014ba53ac0cbb3a8826e).

### Steps to reproduce

Preferably a minimal script to cause the issue to occur.

```python
import numpy as np
from tvm import parser, relay, transform, cpu
from tvm.contrib.graph_executor import GraphModule

code = """
#[version = "0.0.5"]
def @main(%x0: Tensor[(1, 4, 3, 2), float32] /* ty=Tensor[(1, 4, 3, 2), float32] span=from_string:4:15 */, %x1: float32 /* ty=float32 span=from_string:4:20 */, %x2: Tensor[(4), float32] /* ty=Tensor[(4), float32] span=from_string:7:31 */, %x3: Tensor[(4), float32] /* ty=Tensor[(4), float32] span=from_string:7:36 */, %x4: Tensor[(4, 1, 1, 3, 2), float32] /* ty=Tensor[(4, 1, 1, 3, 2), float32] span=from_string:13:24 */, %x5: Tensor[(4), float32] /* ty=Tensor[(4), float32] span=from_string:14:28 */, %x6: Tensor[(4), float32] /* ty=Tensor[(4), float32] span=from_string:14:33 */, %x7: Tensor[(4), float32] /* ty=Tensor[(4), float32] span=from_string:14:38 */, %x8: Tensor[(4), float32] /* ty=Tensor[(4), float32] span=from_string:14:43 */, %x9: Tensor[(4), float32] /* ty=Tensor[(4), float32] span=from_string:16:28 */, %x10: Tensor[(4), float32] /* ty=Tensor[(4), float32] span=from_string:16:33 */, %x11: Tensor[(4), float32] /* ty=Tensor[(4), float32] span=from_string:16:39 */, %x12: Tensor[(4), float32] /* ty=Tensor[(4), float32] span=from_string:16:45 */, %x13: Tensor[(4), float32] /* ty=Tensor[(4), float32] span=from_string:22:33 */, %x14: Tensor[(4), float32] /* ty=Tensor[(4), float32] span=from_string:22:39 */) {
  %0 = nn.pad(%x0, %x1, pad_width=[[0, 0], [0, 0], [2, 2], [1, 0]]) /* ty=Tensor[(1, 4, 7, 3), float32] span=from_string:5:20 */;
  %1 = expand_dims(%0, axis=3) /* ty=Tensor[(1, 4, 7, 1, 3), float32] span=from_string:19:13 */;
  %2 = nn.adaptive_avg_pool3d(%1, output_size=[1, 1, 1]) /* ty=Tensor[(1, 4, 1, 1, 1), float32] span=from_string:8:17 */;
  %3 = nn.instance_norm(%2, %x2, %x3) /* ty=Tensor[(1, 4, 1, 1, 1), float32] span=from_string:8:13 */;
  %4 = (%3, %3, %2) /* ty=(Tensor[(1, 4, 1, 1, 1), float32], Tensor[(1, 4, 1, 1, 1), float32], Tensor[(1, 4, 1, 1, 1), float32]) span=from_string:9:20 */;
  %5 = concatenate(%4, axis=1) /* ty=Tensor[(1, 12, 1, 1, 1), float32] span=from_string:10:12 */;
  %6 = cos(%5) /* ty=Tensor[(1, 12, 1, 1, 1), float32] span=from_string:11:12 */;
  %7 = sum(%6, axis=[1, 2, 3], keepdims=True) /* ty=Tensor[(1, 1, 1, 1, 1), float32] span=from_string:12:15 */;
  %8 = round(%7) /* ty=Tensor[(1, 1, 1, 1, 1), float32] span=from_string:20:23 */;
  %9 = nn.conv3d(%8, %x4, strides=[1, 2, 2], padding=[0, 1, 2, 0, 1, 0], dilation=[1, 1, 2], channels=4, kernel_size=[1, 3, 2]) /* ty=Tensor[(1, 4, 1, 1, 1), float32] span=from_string:14:23 */;
  %10 = nn.batch_norm(%9, %x5, %x6, %x7, %x8, center=False, scale=False) /* ty=(Tensor[(1, 4, 1, 1, 1), float32], Tensor[(4), float32], Tensor[(4), float32]) span=from_string:15:9 */;
  %11 = %10.0 /* ty=Tensor[(1, 4, 1, 1, 1), float32] span=from_string:16:23 */;
  %12 = nn.batch_norm(%11, %x9, %x10, %x11, %x12, center=False, scale=False) /* ty=(Tensor[(1, 4, 1, 1, 1), float32], Tensor[(4), float32], Tensor[(4), float32]) span=from_string:17:9 */;
  %13 = %12.0 /* ty=Tensor[(1, 4, 1, 1, 1), float32] span=from_string:18:13 */;
  %14 = sum(%1, axis=[2], keepdims=True) /* ty=Tensor[(1, 4, 1, 1, 3), float32] span=from_string:20:18 */;
  %15 = exp(%13) /* ty=Tensor[(1, 4, 1, 1, 1), float32] span=from_string:21:13 */;
  %16 = maximum(%14, %8) /* ty=Tensor[(1, 4, 1, 1, 3), float32] span=from_string:21:18 */;
  %17 = add(%15, %16) /* ty=Tensor[(1, 4, 1, 1, 3), float32] span=from_string:22:28 */;
  %18 = nn.instance_norm(%17, %x13, %x14) /* ty=Tensor[(1, 4, 1, 1, 3), float32] span=from_string:23:4 */;
  (%18,) /* ty=(Tensor[(1, 4, 1, 1, 3), float32],) span=from_string:4:3 */
}
"""

mod = parser.parse(code)
inputs = np.load('inputs.npz')
params = np.load('params.npz')

with transform.PassContext(opt_level=0):
    lib = relay.build(mod, params=dict(params), target='llvm')
gmod = GraphModule(lib['default'](cpu()))
gmod.run(**inputs)
ref_out = gmod.get_output(0).numpy()

with transform.PassContext(opt_level=3):
    mod, params = relay.optimize(mod, params=dict(params), target='llvm')
    lib = relay.build(mod, params=params, target='llvm')
gmod = GraphModule(lib['default'](cpu()))
gmod.run(**inputs)
opt_out = gmod.get_output(0).numpy()

print('Reference:\n', ref_out)
print('Optimized:\n', opt_out)
```

### Current Findings

1. This bug occurs in 0.9 and all versions after it, but not in 0.8.
2. This bug seems to be related to `FoldScaleAxis` pass enabled at optimization level 3. If this pass is disabled, the inconsistency will not occur.
3. The test case presented above seems to be minimal to trigger this inconsistency. If any of the operators is removed (i.e., its uses are replaced with the input of the operator), the inconsistency no longer occurs. 
