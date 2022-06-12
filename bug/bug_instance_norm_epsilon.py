# Fixed: https://github.com/apache/tvm/pull/9806

from tvm import relay, transform, IRModule

x = relay.var('x', shape=(1, 4, 8, 8), dtype='float16')
gamma = relay.var('gamma', shape=(4,), dtype='float16')
beta = relay.var('beta', shape=(4,), dtype='float16')
y = relay.nn.instance_norm(x, gamma, beta)
mod = IRModule.from_expr(y)
with transform.PassContext(opt_level=0):
    lib = relay.build(mod, target='llvm')
