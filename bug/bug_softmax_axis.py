# Reported: https://github.com/apache/tvm/issues/11684

from tvm import relay, transform, IRModule

x = relay.var('x', shape=(4,))
y = relay.nn.softmax(x, axis=1)
mod = IRModule.from_expr(y)
mod = relay.transform.InferType()(mod)
with transform.PassContext(opt_level=0):
    lib = relay.build(mod, target='llvm')
