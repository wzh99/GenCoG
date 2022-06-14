# Reported: https://github.com/apache/tvm/issues/11704

from tvm import relay, transform, IRModule

x0 = relay.var('x0', shape=(1, 2))
x1 = relay.var('x1', shape=(1, 2))
x2 = relay.var('x2', shape=(1, 2))
w = relay.var('w', shape=(2, 2))
y0 = relay.maximum(x0, x1)
y1 = relay.nn.dense(x2, w)
y2 = relay.expand_dims(y0, 1)
y3 = y0 * y1
y4 = y2 + y3
mod = IRModule.from_expr(y4)
mod = relay.transform.InferType()(mod)
print(mod)
with transform.PassContext(opt_level=1):
    lib = relay.build(mod, target='llvm')
