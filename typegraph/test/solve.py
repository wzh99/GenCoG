from argparse import ArgumentParser, Namespace
from sys import stdout
from typing import List, cast

from numpy.random import Generator, PCG64
from tqdm import trange
from tvm.parser import parse

from typegraph.expr import TensorType, DataType
from typegraph.expr.ty import ValueType
from typegraph.solve import ConstraintSolver, OpTypeInfo
from typegraph.spec import ConstraintSpec, OpRegistry, max_dim
from typegraph.util import Ref, CodeBuffer

options = Namespace()


def test_all_ops():
    tested_specs = set()
    for name, op in OpRegistry.items():
        spec = op.spec_
        if Ref(spec) in tested_specs:
            continue
        _test_spec(name, spec)
        tested_specs.add(Ref(spec))


def test_one_op(name: str):
    _test_spec(name, OpRegistry.get(name).spec_)


def _test_spec(op: str, spec: ConstraintSpec):
    rng = Generator(PCG64(seed=options.seed))
    for _ in trange(options.iter, file=stdout):
        if spec.has_no_input:
            known = {}
        else:
            rank = rng.choice(spec.first_rank_choices())
            shape = cast(List[int], rng.integers(1, max_dim, rank, endpoint=True).tolist())
            dtype = rng.choice(spec.first_dtype_choices())
            known = {0: TensorType(shape, dtype)}
        solver = ConstraintSolver(spec, known, rng)
        _cmp_relay(op, solver.solve())
    print(f'{op} passed.')


def _cmp_relay(op: str, info: OpTypeInfo):
    # Generate text format and parse to Relay module
    txt = _gen_relay(op, info)
    if options.v:
        print(txt)
    parse(txt)


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
    if len(info.out_types_) > 1 or op in _tuple_out_ops:
        buf.write_pos(map(lambda t: lambda: buf.write(str(t)), info.out_types_))
    else:
        buf.write(str(info.out_types_[0]))

    # Print op
    buf.writeln(' {')
    with buf.indent():
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
        buf.writeln()
    buf.writeln('}')

    return str(buf)


def _fmt_val(v: ValueType):
    if type(v) in (bool, int, float, DataType):
        return str(v)
    elif type(v) is str:
        return '"' + v + '"'
    elif type(v) in (list, tuple):
        return '[' + ', '.join(_fmt_val(e) for e in v) + ']'


def _parse_args():
    global options
    parser = ArgumentParser()
    parser.add_argument('-a', action='store_true')
    parser.add_argument('-v', action='store_true')
    parser.add_argument('-name', type=str)
    parser.add_argument('-iter', type=int)
    parser.add_argument('-seed', type=int, default=42)
    options = parser.parse_args()


if __name__ == '__main__':
    _parse_args()
    if options.a:
        test_all_ops()
    else:
        if options.name is None:
            raise ValueError('Operator name not specified.')
        test_one_op(options.name)
