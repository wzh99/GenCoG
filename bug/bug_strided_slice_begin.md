# [Bug] Index error in type inference of `strided_slice` with empty `begin`

### Expected behavior

The following Relay source program should be successfully parsed:

```
#[version = "0.0.5"]
def @main(%x: Tensor[(4,), float32]) {
  strided_slice(%x, axes=[], begin=[], end=[], strides=[])
}
```

`strided_slice` directly returns a copy of `%x` here because `axes`, `begin`, `end`, and `strides`
are all empty.

### Actual behavior

```
Traceback (most recent call last):
  File "/Users/wzh/tvm-bug/bug_strided_slice_axes.py", line 12, in <module>
    mod = parser.parse(src)
  File "/Users/wzh/tvm-dev/python/tvm/parser/__init__.py", line 32, in parse
    return _ffi_api.ParseModuleInContext(source_name, source, init_module, init_meta_table)
  File "/Users/wzh/tvm-dev/python/tvm/_ffi/_ctypes/packed_func.py", line 237, in __call__
    raise get_last_ffi_error()
tvm._ffi.base.TVMError: Traceback (most recent call last):
  [bt] (8) 9   libtvm.dylib                        0x000000011dea6d79 tvm::parser::ParseModule(std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> > const&, std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> > const&, tvm::runtime::Optional<tvm::IRModule> const&, tvm::runtime::Map<tvm::runtime::String, tvm::runtime::Array<tvm::runtime::ObjectRef, void>, void, void> const&) + 425
  [bt] (7) 8   libtvm.dylib                        0x000000011dd0c2f4 tvm::transform::Pass::operator()(tvm::IRModule) const + 148
  [bt] (6) 7   libtvm.dylib                        0x000000011dd0c701 tvm::transform::Pass::operator()(tvm::IRModule, tvm::transform::PassContext const&) const + 753
  [bt] (5) 6   libtvm.dylib                        0x000000011dd0d203 tvm::transform::ModulePassNode::operator()(tvm::IRModule, tvm::transform::PassContext const&) const + 819
  [bt] (4) 5   libtvm.dylib                        0x000000011f02d05d tvm::runtime::PackedFuncObj::Extractor<tvm::runtime::PackedFuncSubObj<void tvm::runtime::TypedPackedFunc<tvm::IRModule (tvm::IRModule, tvm::transform::PassContext)>::AssignTypedLambda<tvm::relay::transform::InferType()::$_2>(tvm::relay::transform::InferType()::$_2)::'lambda'(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)> >::Call(tvm::runtime::PackedFuncObj const*, tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) + 1933
  [bt] (3) 4   libtvm.dylib                        0x000000011f01c477 tvm::relay::TypeInferencer::Infer(tvm::GlobalVar, tvm::relay::Function) + 135
  [bt] (2) 3   libtvm.dylib                        0x000000011eddfbaf tvm::relay::TypeSolver::Solve() + 1615
  [bt] (1) 2   libtvm.dylib                        0x000000011d9aed29 tvm::runtime::detail::LogFatal::Entry::Finalize() + 89
  [bt] (0) 1   libtvm.dylib                        0x000000011f3b0a28 tvm::runtime::Backtrace() + 24
  [bt] (8) 9   libtvm.dylib                        0x000000011eddf9dc tvm::relay::TypeSolver::Solve() + 1148
  [bt] (7) 8   libtvm.dylib                        0x000000011eddff50 tvm::TypedEnvFunc<bool (tvm::runtime::Array<tvm::Type, void> const&, int, tvm::Attrs const&, tvm::TypeReporter const&)>::operator()(tvm::runtime::Array<tvm::Type, void> const&, int, tvm::Attrs const&, tvm::TypeReporter const&) const + 416
  [bt] (6) 7   libtvm.dylib                        0x000000011de9ccb4 tvm::runtime::PackedFuncObj::Extractor<tvm::runtime::PackedFuncSubObj<void tvm::runtime::TypedPackedFunc<bool (tvm::runtime::Array<tvm::Type, void> const&, int, tvm::Attrs const&, tvm::TypeReporter const&)>::AssignTypedLambda<bool (*)(tvm::runtime::Array<tvm::Type, void> const&, int, tvm::Attrs const&, tvm::TypeReporter const&)>(bool (*)(tvm::runtime::Array<tvm::Type, void> const&, int, tvm::Attrs const&, tvm::TypeReporter const&))::'lambda'(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)> >::Call(tvm::runtime::PackedFuncObj const*, tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*) + 20
  [bt] (5) 6   libtvm.dylib                        0x000000011de9d0c3 void tvm::runtime::TypedPackedFunc<bool (tvm::runtime::Array<tvm::Type, void> const&, int, tvm::Attrs const&, tvm::TypeReporter const&)>::AssignTypedLambda<bool (*)(tvm::runtime::Array<tvm::Type, void> const&, int, tvm::Attrs const&, tvm::TypeReporter const&)>(bool (*)(tvm::runtime::Array<tvm::Type, void> const&, int, tvm::Attrs const&, tvm::TypeReporter const&))::'lambda'(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*)::operator()(tvm::runtime::TVMArgs const&, tvm::runtime::TVMRetValue*) const + 1027
  [bt] (4) 5   libtvm.dylib                        0x000000011ec7e059 tvm::relay::StridedSliceRel(tvm::runtime::Array<tvm::Type, void> const&, int, tvm::Attrs const&, tvm::TypeReporter const&) + 3337
  [bt] (3) 4   libtvm.dylib                        0x000000011ec7eb17 tvm::topi::StridedSliceOutputShape(tvm::runtime::Array<tvm::PrimExpr, void> const&, tvm::runtime::Array<tvm::Integer, void> const&, tvm::runtime::Array<tvm::Integer, void> const&, tvm::runtime::Array<tvm::Integer, void> const&, tvm::runtime::Array<tvm::Integer, void> const&, std::__1::basic_string<char, std::__1::char_traits<char>, std::__1::allocator<char> > const&) + 679
  [bt] (2) 3   libtvm.dylib                        0x000000011dc0bb53 tvm::runtime::Array<tvm::Integer, void>::operator[](long long) const + 403
  [bt] (1) 2   libtvm.dylib                        0x000000011d9aed29 tvm::runtime::detail::LogFatal::Entry::Finalize() + 89
  [bt] (0) 1   libtvm.dylib                        0x000000011f3b0a28 tvm::runtime::Backtrace() + 24
  File "/Users/wzh/tvm-dev/src/relay/analysis/type_solver.cc", line 624
TVMError: 
---------------------------------------------------------------
An error occurred during the execution of TVM.
For more information, please see: https://tvm.apache.org/docs/errors.html
---------------------------------------------------------------
  Check failed: (false) is false: [11:10:08] /Users/wzh/tvm-dev/include/tvm/runtime/container/array.h:393: 
---------------------------------------------------------------
An error occurred during the execution of TVM.
For more information, please see: https://tvm.apache.org/docs/errors.html
---------------------------------------------------------------
  Check failed: (0 <= i && i < p->size_) is false: IndexError: indexing 0 on an array of size 0
```

### Environment

macOS 12.4. Compiled using Clang 13.1.6 with LLVM support. TVM
commit [`df4f4c0b4`](https://github.com/apache/tvm/commit/df4f4c0b4bccd775af25967fdf057392c1a2826e).

### Steps to reproduce

```python
from tvm import parser

src = """
#[version = "0.0.5"]
def @main(%x: Tensor[(4,), float32]) {
    strided_slice(%x, axes=[], begin=[], end=[], strides=[])
}
"""

mod = parser.parse(src)
```
