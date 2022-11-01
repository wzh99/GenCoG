import numpy as np
from tvm import parser, relay, transform, cpu
from tvm.contrib.graph_executor import GraphModule

code = """
#[version = "0.0.5"]
def @main(%x0: Tensor[(14, 1, 11, 7), float16], %x2: Tensor[(5, 1, 3, 1), float16], %x3: Tensor[(5), float16], %x4: Tensor[(5), float16], %x5: Tensor[(5), float16], %x6: Tensor[(5), float16]) -> (Tensor[(14, 5, 5, 3), float16],) {
  %0 = nn.avg_pool2d(%x0, pool_size=[3, 3], strides=[2, 1], dilation=[1, 2], padding=[0, 0, 0, 0], count_include_pad=True) /* from_string */ /* ty=Tensor[(14, 1, 5, 3), float16] */;
  %1 = nn.conv2d(%0, %x2, strides=[1, 2], padding=[1, 1, 1, 2], dilation=[1, 2], channels=5, kernel_size=[3, 1]) /* from_string */ /* ty=Tensor[(14, 5, 5, 3), float16] */;
  %2 = nn.batch_norm(%1, %x3, %x4, %x5, %x6, center=False, scale=False) /* from_string */ /* ty=(Tensor[(14, 5, 5, 3), float16], Tensor[(5), float16], Tensor[(5), float16]) */;
  %3 = %2.0;
  %4 = tan(%3) /* from_string */ /* ty=Tensor[(14, 5, 5, 3), float16] */;
  (%4,)
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

with transform.PassContext(opt_level=3, disabled_pass=['AlterOpLayout']):
    mod, params = relay.optimize(mod, params=params, target='llvm')
    print(mod.astext(show_meta_data=False))
    lib = relay.build(mod, params=dict(params), target='llvm')
gmod = GraphModule(lib['default'](cpu()))
gmod.run(**inputs)
opt_out = gmod.get_output(0).numpy()
diff = np.abs(opt_out - ref_out)
print(f'{np.unravel_index(np.nanargmax(diff), ref_out.shape)}: {np.nanmax(diff)}')
rel_diff = diff / ref_out
print(f'{np.unravel_index(np.nanargmax(rel_diff), ref_out.shape)}: {np.nanmax(rel_diff)}')
