# Fixed: https://github.com/apache/tvm/pull/11325

from tvm import parser

src = """
#[version = "0.0.5"]
def @main(%x: Tensor[(1, 2, 4), float32], %w: Tensor[(4, 2, 1), float32]) {
    nn.conv1d(%x, %w, kernel_size=[1], padding=[1, 1], out_layout="NCW")
}
"""

mod = parser.parse(src)
print(mod)
