# Fixed

from tvm import relay, transform, IRModule

x = relay.var('x', shape=(1, 4), dtype='int32')
y = relay.exp(x)
mod = IRModule.from_expr(y)
print(mod)
mod = relay.transform.InferType()(mod)
with transform.PassContext(opt_level=0):
    lib = relay.build(mod, target='llvm')
