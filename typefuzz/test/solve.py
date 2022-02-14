from argparse import ArgumentParser, Namespace
from os import environ
from sys import stdout
from typing import List, cast

from numpy.random import Generator, PCG64
from tqdm import trange
from tvm import parser, relay, transform, device
from tvm.contrib.graph_executor import GraphModule

from typefuzz.expr import TensorType, DataType
from typefuzz.expr.ty import ValueType
from typefuzz.solve import TypeSolver, OpTypeInfo
from typefuzz.spec import TypeSpec, OpRegistry, max_dim
from typefuzz.util import Ref, CodeBuffer, run_process

options = Namespace()


def test_all_ops():
    tested_specs = set()
    for name, op in OpRegistry.items():
        spec = op.spec
        if Ref(spec) in tested_specs:
            print(f'{op} specification tested before.')
            continue
        _test_spec(name, spec)
        tested_specs.add(Ref(spec))


def test_one_op(name: str):
    _test_spec(name, OpRegistry.get(name).spec)


def _test_spec(op: str, spec: TypeSpec):
    rng = Generator(PCG64(seed=options.seed))
    for _ in trange(options.iter, file=stdout):
        if spec.has_no_input:
            known = {}
        else:
            rank = rng.choice(spec.first_rank_choices())
            shape = cast(List[int], rng.integers(1, max_dim, rank, endpoint=True).tolist())
            dtype = rng.choice(spec.first_dtype_choices())
            known = {0: TensorType(shape, dtype)}
        solver = TypeSolver(spec, known, rng)
        _compile_relay(op, solver.solve())
    print(f'{op} passed.')


def _compile_relay(op: str, info: OpTypeInfo):
    src = _gen_relay(op, info)
    if options.verbose:
        print(src)
    if options.separate:
        result = run_process(_compile_func, (src,))
        if options.verbose and result.exitcode != 0:
            print(f'Compilation error: Exit code {result.exitcode}.')
    else:
        _compile_func(src)


def _compile_func(src: str):
    mod = parser.parse(src)
    with transform.PassContext(opt_level=3):
        lib = relay.build(mod, 'llvm')
    dev = device('cpu', 0)
    GraphModule(lib['default'](dev))
    return dict()


_tuple_in_ops = {
    'concatenate',
}

_tuple_out_ops = {
    'split',
}


def _gen_relay(op: str, info: OpTypeInfo):
    # Prelude
    buf = CodeBuffer()
    buf.writeln('#[version = "0.0.5"]')
    buf.write('def @main')

    # Tensor types
    buf.write_pos(
        map(lambda p: lambda: buf.write(f'%x{p[0]}: {p[1]}'), enumerate(info.in_types_))
    )
    buf.write(' -> ')
    buf.write(str(info.out_types_[0]))

    # Print op
    buf.writeln(' {')
    with buf.indent():
        buf.write('%0 = ')
        buf.write(op)
        args = map(lambda i: f'%x{i}', range(len(info.in_types_)))
        if op in _tuple_in_ops:
            arg_str = str(tuple(args)).replace('\'', '')
        else:
            arg_str = ', '.join(args)
        buf.write_pos([
            lambda: buf.write(arg_str),
            lambda: buf.write_named(
                map(lambda a: (a[0], lambda: buf.write(_fmt_val(a[1]))), info.attrs_),
                prefix='', suffix=''
            )
        ])
        buf.writeln(';')
        buf.write('%0')
        if len(info.out_types_) > 1 or op in _tuple_out_ops:
            buf.write('.0')
        buf.writeln()
    buf.writeln('}')

    return str(buf)


def _fmt_val(v: ValueType):
    if isinstance(v, (bool, int, float, DataType)):
        return str(v)
    elif isinstance(v, str):
        return '"' + v + '"'
    elif isinstance(v, (tuple, list)):
        return '[' + ', '.join(_fmt_val(e) for e in v) + ']'


def parse_args():
    global options
    p = ArgumentParser()
    p.add_argument('-a', '--all', action='store_true', help='Test all operators.')
    p.add_argument('-v', '--verbose', action='store_true', help='Run in verbose mode.')
    p.add_argument('-s', '--separate', action='store_true',
                   help='Compile module in a separate process.')
    p.add_argument('-name', type=str, help='Name of the operator to be tested.')
    p.add_argument('-iter', type=int, help='Iteration number of each operator.')
    p.add_argument('-seed', type=int, default=42, help='Random seed of test case generator.')
    options = p.parse_args()
    if not options.separate:
        environ['TVM_BACKTRACE'] = '1'


if __name__ == '__main__':
    parse_args()
    if options.all:
        test_all_ops()
    else:
        if options.name is None:
            raise ValueError('Operator name not specified.')
        test_one_op(options.name)
