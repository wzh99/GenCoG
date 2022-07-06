# Fixed: https://github.com/apache/tvm/pull/11681

from tvm import relay, IRModule

src = """
#[version = "0.0.5"]
def @main(%x: Tensor[(1, 16, 32, 32, 32), float32], %w: Tensor[(4, 4, 1, 1, 1), float32]) {
    nn.conv3d(%x, %w, kernel_size=[1, 1, 1], channels=4, groups=4)
}
"""

x = relay.var('x', shape=(1, 16, 32, 32, 32))
w = relay.var('w', shape=(4, 4, 1, 1, 1))
y = relay.nn.conv3d(x, w, kernel_size=(1, 1, 1), channels=4, groups=4)
mod = IRModule.from_expr(y)
mod = relay.transform.InferType()(mod)
print(mod)
