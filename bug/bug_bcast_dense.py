# Confirmed: https://github.com/apache/tvm/issues/11704

from tvm import relay, transform, IRModule

x0 = relay.var('x0', shape=(1, 2))
x1 = relay.var('x1', shape=(1, 2))
x2 = relay.var('x2', shape=(2, 2))
w = relay.var('w', shape=(2, 2))
y0 = relay.maximum(x0, x1)
y1 = relay.nn.dense(x2, w)
y2 = relay.expand_dims(y0, 1)
y3 = y0 * y1
y4 = y2 + y3
# y3 = y1 * y0
# y4 = y3 + y2
mod = IRModule.from_expr(y4)
mod = relay.transform.InferType()(mod)
print(mod)
mod = relay.transform.FuseOps()(mod)
print(mod)
with transform.PassContext(opt_level=1):
    lib = relay.build(mod, target='llvm')
