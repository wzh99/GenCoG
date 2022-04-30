import os.path
from enum import IntEnum, auto
from typing import Dict

import numpy as np
from numpy.random import Generator
from tvm import transform, relay, parser, cpu, TVMError, IRModule
from tvm.contrib.graph_executor import GraphModule


class ErrorKind(IntEnum):
    PARSE = auto()
    COMPILE = auto()
    RUN = auto()
    COMPUTE = auto()


class RelayRunner:
    def __init__(self, rng: Generator, path: str):
        self._rng = rng
        self._path = path

    def run(self, src: str) -> bool:
        # Parse module
        try:
            mod = parser.parse(src)
        except TVMError as err:
            self.write_report(ErrorKind.PARSE, src, str(err), 0)
            return False

        # Generate input parameters
        main_fn = mod['main']
        inputs = {main_fn.params[0].name_hint: gen_tensor_value(main_fn.params[0], self._rng)}
        params = {var.name_hint: gen_tensor_value(var, self._rng) for var in main_fn.params[1:]}

        # Build and run unoptimized module as reference
        try:
            gmod = build_mod(mod, params, 0)
        except Exception as err:
            self.write_report(ErrorKind.COMPILE, mod.astext(), str(err), 0)
            return False
        try:
            ref_outputs = run_exec(gmod, inputs)
        except Exception as err:
            self.write_report(ErrorKind.RUN, mod.astext(), str(err), 0)
            return False

        # Build and run modules with different levels of optimization
        for opt_level in range(1, 4):
            try:
                gmod = build_mod(mod, params, opt_level)
            except Exception as err:
                self.write_report(ErrorKind.COMPILE, mod.astext(), str(err), opt_level)
                return False
            try:
                outputs = run_exec(gmod, inputs)
            except Exception as err:
                self.write_report(ErrorKind.RUN, mod.astext(), str(err), opt_level)
                return False
            for i, (o, ro) in enumerate(zip(outputs, ref_outputs)):
                if not np.allclose(o, ro, rtol=1e-2, atol=1e-3, equal_nan=True):
                    msg = f'Computation error in output tensor {i}:\n' \
                          f'Expect:\n' \
                          f'{np.array_repr(ro)}\n' \
                          f'Actual:\n' \
                          f'{np.array_repr(o)}'
                    self.write_report(ErrorKind.COMPUTE, mod.astext(), msg, opt_level)
                    return False

        return True

    def write_report(self, kind: ErrorKind, code: str, err: str, opt_level: int):
        if not os.path.exists(self._path):
            os.mkdir(self._path)
        with open(os.path.join(self._path, 'kind.txt'), 'w') as file:
            file.write(kind.name)
        with open(os.path.join(self._path, 'code.txt'), 'w') as file:
            file.write(code)
        with open(os.path.join(self._path, 'error.txt'), 'w') as file:
            file.write(f'At optimization level {opt_level}:\n')
            file.write(err)


def gen_tensor_value(var: relay.Var, rng: Generator):
    var_ty = var.checked_type
    return rng.standard_normal(
        size=[int(d) for d in var_ty.shape], dtype='float64'
    ).astype(var_ty.dtype)


def build_mod(mod: IRModule, params: Dict[str, np.ndarray], opt_level: int):
    with transform.PassContext(opt_level=opt_level):
        lib = relay.build(mod, target='llvm', params=params)
    return GraphModule(lib['default'](cpu()))


def run_exec(gmod: GraphModule, inputs: Dict[str, np.ndarray]):
    gmod.run(**inputs)
    return [gmod.get_output(i).numpy() for i in range(gmod.get_num_outputs())]
