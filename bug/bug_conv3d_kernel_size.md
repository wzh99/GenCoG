# [Bug] Weight shape inferred for `nn.conv3d` is wrong when `kernel_size` and `channels` are both provided

### Expected behavior

The following Relay source program should be successfully parsed:

```
#[version = "0.0.5"]
def @main(%x: Tensor[(1, 4, 8, 8, 8), float32], %w: Tensor[(2, 2, 1, 1, 1), float32]) {
    nn.conv3d(%x, %w, kernel_size=[1, 1, 1], channels=2, groups=2)
}
```

Here is the explanation: 4 input channels are divided into 2 groups, with each group taking 2
channels as input. Each group produces 1 output channel. Therefore, there are 2 * 1 = 2 output
channels in total.

### Actual behavior

The Relay type checker reports a type mismatch:

```
error: The Relay type checker is unable to show the following types match:
  Tensor[(4, 0, 1, 1, 1), float32]
  Tensor[(2, 2, 1, 1, 1), float32]
In particular:
  dimension 0 conflicts: 4 does not match 2.  dimension 1 conflicts: 0 does not match 2.
 --> from_string:4:5
   |  
 4 |      nn.conv3d(%x, %w, kernel_size=[1, 1, 1], channels=2, groups=2)
   |      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
error: The Relay type checker is unable to show the following types match.
In particular `Tensor[(2, 2, 1, 1, 1), float32]` does not match `Tensor[(4, 0, 1, 1, 1), float32]`
 --> from_string:4:5
   |  
 4 |      nn.conv3d(%x, %w, kernel_size=[1, 1, 1], channels=2, groups=2)
   |      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Traceback (most recent call last):
  File "/Users/wzh/tvm-bug/bug_conv3d_kernel_size.py", line 12, in <module>
    mod = parser.parse(src)
  File "/Users/wzh/tvm-dev/python/tvm/parser/__init__.py", line 32, in parse
    return _ffi_api.ParseModuleInContext(source_name, source, init_module, init_meta_table)
  File "/Users/wzh/tvm-dev/python/tvm/_ffi/_ctypes/packed_func.py", line 237, in __call__
    raise get_last_ffi_error()
tvm.error.DiagnosticError: Traceback (most recent call last):
  [bt] (8) 9   libtvm.dylib                        0x0000000118057102 tvm::runtime::PackedFuncObj::Extractor<tvm::runtime::PackedFuncSubObj<void tvm::runtime::TypedPackedFunc<tvm::IRModule (std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> > const&, std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> > const&, tvm::runtime::Optional<tvm::IRModule> const&, tvm::runtime::Map<tvm::runtime::String, tvm::runtime::Array<tvm::runtime::ObjectRef, void>, void, void> const&)>::AssignTypedLambda<tvm::parser::$_2>(tvm::parser::$_2, std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> >)::'lambda'(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)> >::Call(tvm::runtime::PackedFuncObj const*, tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) + 1090
  [bt] (7) 8   libtvm.dylib                        0x0000000118023d79 tvm::parser::ParseModule(std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> > const&, std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> > const&, tvm::runtime::Optional<tvm::IRModule> const&, tvm::runtime::Map<tvm::runtime::String, tvm::runtime::Array<tvm::runtime::ObjectRef, void>, void, void> const&) + 425
  [bt] (6) 7   libtvm.dylib                        0x0000000117e892f4 tvm::transform::Pass::operator()(tvm::IRModule) const + 148
  [bt] (5) 6   libtvm.dylib                        0x0000000117e89701 tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const + 753
  [bt] (4) 5   libtvm.dylib                        0x0000000117e8a203 tvm::transform::ModulePassNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const + 819
  [bt] (3) 4   libtvm.dylib                        0x00000001191aa0d5 tvm::runtime::PackedFuncObj::Extractor<tvm::runtime::PackedFuncSubObj<void tvm::runtime::TypedPackedFunc<tvm::IRModule (tvm::IRModule, tvm::transform::PassContext)>::AssignTypedLambda<tvm::relay::transform::InferType()::$_2>(tvm::relay::transform::InferType()::$_2)::'lambda'(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)> >::Call(tvm::runtime::PackedFuncObj const*, tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) + 2053
  [bt] (2) 3   libtvm.dylib                        0x0000000117e0da1b tvm::DiagnosticContext::Render() + 459
  [bt] (1) 2   libtvm.dylib                        0x0000000117b2bd29 tvm::runtime::detail::LogFatal::Entry::Finalize() + 89
  [bt] (0) 1   libtvm.dylib                        0x000000011952da28 tvm::runtime::Backtrace() + 24
  File "/Users/wzh/tvm-dev/src/ir/diagnostic.cc", line 105
DiagnosticError: one or more error diagnostics were emitted, please check diagnostic render for output.
```

### Environment

macOS 12.4. Compiled using Clang 13.1.6 with LLVM support. TVM
commit [`df4f4c0b4`](https://github.com/apache/tvm/commit/df4f4c0b4bccd775af25967fdf057392c1a2826e).

### Steps to reproduce

```python
from tvm import parser

src = """
#[version = "0.0.5"]
def @main(%x: Tensor[(1, 4, 8, 8, 8), float32], %w: Tensor[(2, 2, 1, 1, 1), float32]) {
    nn.conv3d(%x, %w, kernel_size=[1, 1, 1], channels=2, groups=2)
}
"""

mod = parser.parse(src)
```

### Possible fix

[tvm/src/relay/op/nn/convolution.cc#L444-L451](https://github.com/apache/tvm/blob/df4f4c0b4bccd775af25967fdf057392c1a2826e/src/relay/op/nn/convolution.cc#L444-L451)

The condition of this if statement seems to be incorrect. `channels == groups && channels != 1` does
not necessarily mean that the convolution is depthwise. The test case shown above is a
counterexample.

A possible fix is to remove this if statement and only keeps the assignment to `wshape` in else
branch, since this assignment already covers the case of depthwise convolution. 
