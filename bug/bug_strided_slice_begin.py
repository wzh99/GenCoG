# Reported: https://github.com/apache/tvm/issues/11679

from tvm import parser

src = """
#[version = "0.0.5"]
def @main(%x: Tensor[(4,), float32]) {
    strided_slice(%x, begin=[], end=[], strides=[], axes=[])
}
"""

mod = parser.parse(src)
