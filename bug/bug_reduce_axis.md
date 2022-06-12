# [Bug] Segmentation fault for type inference of reduction operators with `axis=[]`

Type inference of the following Relay function causes a segmentation fault:

```
def @main(%x: Tensor[(1, 2, 3), float32]) {
  sum(%x, axis=[], keepdims=True, exclude=True)
}
```

### Expected behavior

The type inference pass accepts this function, since the type constraints of the reduction
operator `sum` are not violated.

### Actual behavior

Segmentation fault.

### Environment

macOS 12.4. Compiled using Clang 13.1.6 with LLVM support. TVM
commit [`df4f4c0b4`](https://github.com/apache/tvm/commit/df4f4c0b4bccd775af25967fdf057392c1a2826e).

### Steps to reproduce

```python
from tvm import relay, ir

x = relay.var('x', shape=(1, 2, 3))
y = relay.sum(x, axis=(), keepdims=True, exclude=True)
mod = ir.IRModule.from_expr(y)
mod = relay.transform.InferType()(mod)
```

### Possible fix

[tvm/src/relay/op/tensor/reduce.cc#L72-L73](https://github.com/apache/tvm/blob/df4f4c0b4bccd775af25967fdf057392c1a2826e/src/relay/op/tensor/reduce.cc#L72-L73)

When `in_axes` is empty, accessing it by index `in_axes.size() - 1` is invalid. To fix it, I suggest
removing this `ICHECK` because the range of elements in `in_axes` seems to have been checked already
in the previous for-loop. 