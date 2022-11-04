# Fixed: https://github.com/apache/tvm/pull/13280

from tvm import relay, IRModule

x = relay.var('x', shape=(1, 4))
gamma = relay.var('gamma', shape=(4,))
beta = relay.var('beta', shape=(4,))
y = relay.nn.instance_norm(x, gamma, beta)
mod = IRModule.from_expr(y)
print(mod)
mod = relay.transform.InferType()(mod)
mod = relay.transform.SimplifyInference()(mod)
print(mod)
