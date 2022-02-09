if __name__ == '__main__':
    from tvm import relay, ir

    x = relay.var('x', shape=(12, 100, 84))
    w = relay.var('w', shape=(56, 84))
    y = relay.nn.dense(x, w)
    f = relay.Function(relay.analysis.free_vars(y), y)  # Tensor[(61, 24, 97, 122)
    mod = ir.IRModule.from_expr(f)
    mod = relay.transform.InferType()(mod)
    print(mod)
    pass

"""
def @main(%x: Tensor[(11, 118, 110, 62, 99), float32], %w: Tensor[(59, 2, 5, 7, 1), float32]) -> Tensor[(11, 59, 116, 27, 53), float32] {
  nn.conv3d(%x, %w, strides=[1, 2, 2], padding=[3, 3, 2, 7, 7, 4], dilation=[1, 3, 1], groups=59) /* ty=Tensor[(11, 59, 116, 27, 53), float32] */
}
"""
