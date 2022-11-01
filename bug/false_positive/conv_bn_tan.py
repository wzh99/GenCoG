import numpy as np
from tvm import parser, relay, transform, cpu
from tvm.contrib.graph_executor import GraphModule

code = """
#[version = "0.0.5"]
def @main(%x0: Tensor[(14, 1, 11, 7), float16], %x2: Tensor[(5, 1, 3, 1), float16], 
    %x3: Tensor[(5), float16], %x4: Tensor[(5), float16], %x5: Tensor[(5), float16], 
    %x6: Tensor[(5), float16]) -> Tensor[(14, 5, 5, 4), float16] {
  %0 = nn.conv2d(%x0, %x2, strides=[2, 2], padding=[0, 0, 0, 0], dilation=[1, 1]);
  %1 = nn.batch_norm(%0, %x3, %x4, %x5, %x6, axis=1, center=False, scale=False);
  tan(%1.0)
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
max_idx = np.unravel_index(np.nanargmax(diff), ref_out.shape)
print('pos:', max_idx)
ref_elem, opt_elem = ref_out[max_idx], opt_out[max_idx]
print('ref:', ref_elem, 'opt:', opt_elem)
print('abs_err:', np.abs(ref_elem - opt_elem), 'rel_err:', np.abs(ref_elem - opt_elem) / ref_elem)
