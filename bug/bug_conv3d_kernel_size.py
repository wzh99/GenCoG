# Confirmed: https://github.com/apache/tvm/issues/11664

from tvm import parser

src = """
#[version = "0.0.5"]
def @main(%x: Tensor[(1, 16, 32, 32, 32), float32], %w: Tensor[(4, 4, 1, 1, 1), float32]) {
    nn.conv3d(%x, %w, kernel_size=[1, 1, 1], channels=4, groups=4)
}
"""

mod = parser.parse(src)
print(mod)
