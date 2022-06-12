# [Bug] Type inference of `nn.softmax` does not reject invalid `axis`

### Expected behavior

The following Relay program should NOT pass type inference:

```
#[version = "0.0.5"]
def @main(%x: Tensor[(4), float32]) {
  nn.softmax(%x, axis=1)
}
```

The input tensor `%x` of `nn.softmax` has only one dimension. The valid range of `axis` is [-1, 1)
. `axis=1` is obviously invalid in this case.

### Actual behavior

This program passes type inference of Relay.

### Environment

macOS 12.4. Compiled using Clang 13.1.6 with LLVM support. TVM
commit [`0df6961`](https://github.com/apache/tvm/commit/0df69611b2fb46724a0023dd8d389c9a1ecedcb8).

### Steps to reproduce

```python
from tvm import relay, IRModule

x = relay.var('x', shape=(4,))
y = relay.nn.softmax(x, axis=1)
mod = IRModule.from_expr(y)
mod = relay.transform.InferType()(mod)
```

### Possible fix

[tvm/src/relay/op/nn/nn.cc#L409-L423](https://github.com/apache/tvm/blob/0df69611b2fb46724a0023dd8d389c9a1ecedcb8/src/relay/op/nn/nn.cc#L409-L423)

In operator registration of `nn.softmax`, its type relation is set to be `IdentityRel`.
However, `nn.softmax` has an attribute `axis` that is not checked by `IdentityRel`.

A possible fix is to implement a new type relation function that checks `axis` attribute
in `SoftmaxAttrs`. This type relation also applies to `nn.fast_softmax` and `nn.log_softmax`. 
