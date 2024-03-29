# Fixed: https://github.com/apache/tvm/pull/11643

from tvm import relay, ir

x = relay.var('x', shape=(1, 2, 3))
y = relay.sum(x, axis=())
mod = ir.IRModule.from_expr(y)
mod = relay.transform.InferType()(mod)
