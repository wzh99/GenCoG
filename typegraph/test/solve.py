from argparse import ArgumentParser, Namespace
from typing import List, cast

import numpy as np
from numpy.random import Generator, PCG64

from typegraph.expr import TensorType, DataType
from typegraph.solve import ConstraintSolver
from typegraph.spec import ConstraintSpec, OpRegistry
from typegraph.util import Ref

options = Namespace()


def test_all_ops():
    tested_specs = set()
    for _, op in OpRegistry.items():
        spec = op.spec_
        if Ref(spec) in tested_specs:
            continue
        _test_spec(spec)
        tested_specs.add(Ref(spec))


def test_one_op(name: str):
    _test_spec(OpRegistry.get(name).spec_)


def _test_spec(spec: ConstraintSpec):
    rng = Generator(PCG64(seed=options.seed))
    for _ in range(options.iter):
        shape = cast(List[int], np.concatenate([
            [rng.integers(1, 16, endpoint=True)],
            rng.integers(1, 128, 3, endpoint=True)
        ]).tolist())
        known = {0: TensorType(shape, DataType.f(32))}
        solver = ConstraintSolver(spec, known, rng)
        print(solver.solve())


def _parse_args():
    global options
    parser = ArgumentParser()
    parser.add_argument('-a')
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
