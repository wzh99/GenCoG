# [Bug] Relay program with broadcasting operators and `dense` cannot be compiled

### Expected behavior

The following Relay program should be successfully compiled:

```
def @main(%x0: Tensor[(1, 2), float32] /* ty=Tensor[(1, 2), float32] */, %x1: Tensor[(1, 2), float32] /* ty=Tensor[(1, 2), float32] */, %x2: Tensor[(1, 2), float32] /* ty=Tensor[(1, 2), float32] */, %w: Tensor[(2, 2), float32] /* ty=Tensor[(2, 2), float32] */) -> Tensor[(1, 1, 2), float32] {
  %0 = maximum(%x0, %x1) /* ty=Tensor[(1, 2), float32] */;
  %1 = nn.dense(%x2, %w, units=None) /* ty=Tensor[(1, 2), float32] */;
  %2 = expand_dims(%0, axis=1) /* ty=Tensor[(1, 1, 2), float32] */;
  %3 = multiply(%0, %1) /* ty=Tensor[(1, 2), float32] */;
  add(%2, %3) /* ty=Tensor[(1, 1, 2), float32] */
}
```

### Actual behavior

An error is reported during compilation:

```
Traceback (most recent call last):
  File "/Users/wzh/tvm-bug/bug_dense_bcast.py", line 16, in <module>
    lib = relay.build(mod, target='llvm')
  File "/Users/wzh/tvm-dev/python/tvm/relay/build_module.py", line 416, in build
    graph_json, runtime_mod, params = bld_mod.build(
  File "/Users/wzh/tvm-dev/python/tvm/relay/build_module.py", line 154, in build
    self._build(mod, raw_targets, executor, runtime, workspace_memory_pools, mod_name)
  File "/Users/wzh/tvm-dev/python/tvm/_ffi/_ctypes/packed_func.py", line 237, in __call__
    raise get_last_ffi_error()
ValueError: Traceback (most recent call last):
  [bt] (8) 9   libtvm.dylib                        0x000000011400e502 tvm::relay::tec::LowerTensorExprMutator::DeviceAwareVisitExpr_(tvm::relay::CallNode const*) + 4578
  [bt] (7) 8   libtvm.dylib                        0x0000000113ffd163 tvm::relay::tec::TECompilerImpl::Lower(tvm::relay::tec::CCacheKey const&, tvm::runtime::String) + 131
  [bt] (6) 7   libtvm.dylib                        0x0000000113ffd02e tvm::relay::tec::TECompilerImpl::Lower(tvm::relay::tec::CCacheKey const&, std::__1::function<tvm::runtime::String (tvm::runtime::String)>) + 110
  [bt] (5) 6   libtvm.dylib                        0x00000001140018e4 tvm::relay::tec::TECompilerImpl::LowerInternal(tvm::relay::tec::CCacheKey const&, std::__1::function<tvm::runtime::String (tvm::runtime::String)>) + 2596
  [bt] (4) 5   libtvm.dylib                        0x000000011401884d tvm::relay::tec::PrimFuncFor(tvm::relay::Function const&, tvm::Target const&, std::__1::function<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> > (std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> >)>) + 141
  [bt] (3) 4   libtvm.dylib                        0x000000011401a626 tvm::relay::tec::ScheduleBuilder::Create(tvm::relay::Function const&, std::__1::function<std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> > (std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> >)>) + 7414
  [bt] (2) 3   libtvm.dylib                        0x00000001140f249a tvm::relay::OpImplementation::Schedule(tvm::Attrs const&, tvm::runtime::Array<tvm::te::Tensor, void> const&, tvm::Target const&) + 202
  [bt] (1) 2   libtvm.dylib                        0x00000001142c0b0d tvm::runtime::PackedFuncObj::Extractor<tvm::runtime::PackedFuncSubObj<TVMFuncCreateFromCFunc::$_2> >::Call(tvm::runtime::PackedFuncObj const*, tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) + 109
  [bt] (0) 1   libtvm.dylib                        0x00000001142db328 tvm::runtime::Backtrace() + 24
  File "/Users/wzh/tvm-dev/python/tvm/_ffi/_ctypes/packed_func.py", line 81, in cfun
    rv = local_pyfunc(*pyargs)
  File "/Users/wzh/tvm-dev/python/tvm/relay/op/strategy/generic.py", line 51, in wrapper
    return topi_schedule(outs)
  File "/Users/wzh/tvm-dev/python/tvm/autotvm/task/topi_integration.py", line 242, in wrapper
    return topi_schedule(cfg, outs, *args, **kwargs)
  File "/Users/wzh/tvm-dev/python/tvm/topi/x86/dense.py", line 279, in schedule_dense_pack
    traverse_inline(s, outs[0].op, _callback)
  File "/Users/wzh/tvm-dev/python/tvm/topi/utils.py", line 81, in traverse_inline
    _traverse(final_op)
  File "/Users/wzh/tvm-dev/python/tvm/topi/utils.py", line 78, in _traverse
    _traverse(tensor.op)
  File "/Users/wzh/tvm-dev/python/tvm/topi/utils.py", line 78, in _traverse
    _traverse(tensor.op)
  File "/Users/wzh/tvm-dev/python/tvm/topi/utils.py", line 79, in _traverse
    callback(op)
  File "/Users/wzh/tvm-dev/python/tvm/topi/x86/dense.py", line 277, in _callback
    _schedule_dense_pack_template(cfg, s, op.output(0), outs[0])
  File "/Users/wzh/tvm-dev/python/tvm/topi/x86/dense.py", line 70, in _schedule_dense_pack_template
    y, x = s[O].op.axis
ValueError: too many values to unpack (expected 2)
```

### Environment

macOS 12.4. Compiled using Clang 13.1.6 with LLVM support. TVM
commit [`8341e33`](https://github.com/apache/tvm/commit/8341e33d05868b7bb8496c913679b7951836f3b9).

### Steps to reproduce

```python
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
```
