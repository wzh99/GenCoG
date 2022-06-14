# [Bug] Execution error of a simple `nn.bias_add` + `squeeze` graph where `axis=[]`

### Expected behavior

The following Relay program should be successfully executed:

```
def @main(%x: Tensor[(1, 4), float32] /* ty=Tensor[(1, 4), float32] */, %b: Tensor[(4), float32] /* ty=Tensor[(4), float32] */) -> Tensor[(1, 4), float32] {
  %0 = nn.bias_add(%x, %b) /* ty=Tensor[(1, 4), float32] */;
  squeeze(%0, axis=[]) /* ty=Tensor[(1, 4), float32] */
}
```

### Actual behavior

An error is reported during graph execution:

```
Traceback (most recent call last):
  File "/Users/wzh/tvm-bug/bug_squeeze_axis.py", line 21, in <module>
    gmod.run(x=x_input, b=b_input)
  File "/Users/wzh/tvm-dev/python/tvm/contrib/graph_executor.py", line 208, in run
    self._run()
  File "/Users/wzh/tvm-dev/python/tvm/_ffi/_ctypes/packed_func.py", line 237, in __call__
    raise get_last_ffi_error()
tvm._ffi.base.TVMError: Traceback (most recent call last):
  [bt] (7) 8   ???                                 0x00007ff7bbc612f0 0x0 + 140701983970032
  [bt] (6) 7   _ctypes.cpython-38-darwin.so        0x000000010458dfb7 ffi_call_unix64 + 79
  [bt] (5) 6   libtvm.dylib                        0x0000000116eea24e TVMFuncCall + 62
  [bt] (4) 5   libtvm.dylib                        0x0000000116fbf2ca tvm::runtime::PackedFuncObj::Extractor<tvm::runtime::PackedFuncSubObj<tvm::runtime::GraphExecutor::GetFunction(std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> > const&, tvm::runtime::ObjectPtr<tvm::runtime::Object> const&)::$_12> >::Call(tvm::runtime::PackedFuncObj const*, tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) + 106
  [bt] (3) 4   libtvm.dylib                        0x0000000116fbd331 std::__1::__function::__func<tvm::runtime::GraphExecutor::CreateTVMOp(tvm::runtime::TVMOpParam const&, std::__1::vector<DLTensor, std::__1::allocator<DLTensor> > const&)::$_2, std::__1::allocator<tvm::runtime::GraphExecutor::CreateTVMOp(tvm::runtime::TVMOpParam const&, std::__1::vector<DLTensor, std::__1::allocator<DLTensor> > const&)::$_2>, void ()>::operator()() + 81
  [bt] (2) 3   libtvm.dylib                        0x0000000116f0625d tvm::runtime::PackedFuncObj::Extractor<tvm::runtime::PackedFuncSubObj<tvm::runtime::WrapPackedFunc(int (*)(TVMValue*, int*, int, TVMValue*, int*, void*), tvm::runtime::ObjectPtr<tvm::runtime::Object> const&)::$_0> >::Call(tvm::runtime::PackedFuncObj const*, tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) + 397
  [bt] (1) 2   libtvm.dylib                        0x00000001154f9fc9 tvm::runtime::detail::LogFatal::Entry::Finalize() + 89
  [bt] (0) 1   libtvm.dylib                        0x0000000116f07328 tvm::runtime::Backtrace() + 24
  File "/Users/wzh/tvm-dev/src/runtime/library_module.cc", line 80
TVMError: 
---------------------------------------------------------------
An error occurred during the execution of TVM.
For more information, please see: https://tvm.apache.org/docs/errors.html
---------------------------------------------------------------

  Check failed: ret == 0 (-1 vs. 0) : Assert fail: (1 == tir.tvm_struct_get(arg.T_squeeze, 0, 4)), arg.T_squeeze.ndim is expected to equal 1
```

### Environment

macOS 12.4. Compiled using Clang 13.1.6 with LLVM support. TVM
commit [`8341e33`](https://github.com/apache/tvm/commit/8341e33d05868b7bb8496c913679b7951836f3b9).

### Steps to reproduce

```python
import numpy as np
from tvm import relay, transform, cpu, IRModule
from tvm.contrib.graph_executor import GraphModule

x_shape = (1, 4)
b_shape = (4,)

x = relay.var('x', shape=x_shape)
b = relay.var('b', shape=b_shape)
y = relay.nn.bias_add(x, b)
y = relay.squeeze(y, axis=[])
mod = IRModule.from_expr(y)
with transform.PassContext(opt_level=1):
    lib = relay.build(mod, target='llvm')
gmod = GraphModule(lib['default'](cpu()))

x_input = np.zeros(shape=x_shape, dtype='float32')
b_input = np.zeros(shape=b_shape, dtype='float32')
gmod.run(x=x_input, b=b_input)
```
