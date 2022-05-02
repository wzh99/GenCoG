import os.path
from enum import IntEnum, auto
from typing import Dict, Optional, List

import numpy as np
from numpy.random import Generator
from tvm import transform, relay, parser, cpu, TVMError, IRModule
from tvm.contrib.graph_executor import GraphModule


class ErrorKind(IntEnum):
    PARSE = auto()
    COMPILE = auto()
    RUN = auto()
    COMPUTE = auto()


class ModuleError(Exception):
    def __init__(self, kind: ErrorKind, code: str, err: str, opt_level: int):
        self.kind_ = kind
        self.code_ = code
        self.err_ = err
        self.opt_level_ = opt_level

    def write_txt(self, path: str):
        if not os.path.exists(path):
            os.mkdir(path)
        with open(os.path.join(path, self.kind_.name), 'w'):
            pass
        with open(os.path.join(path, 'code.txt'), 'w') as file:
            file.write(self.code_)
        with open(os.path.join(path, 'error.txt'), 'w') as file:
            file.write(f'opt_level={self.opt_level_}\n')
            file.write(self.err_)


class ModuleRunner:
    def __init__(self, rng: Generator):
        self._rng = rng

    def run(self, code: str):
        # Parse module
        try:
            mod = parser.parse(code)
        except TVMError as err:
            raise ModuleError(ErrorKind.PARSE, code, str(err), 0)

        # Generate input parameters
        main_fn = mod['main']
        inputs = gen_tensor_value_dict(main_fn.params[0:1], self._rng)
        params = gen_tensor_value_dict(main_fn.params[1:], self._rng)

        # Build and run unoptimized module as reference
        try:
            gmod = build_mod(mod, 0, params=params)
        except Exception as err:
            raise ModuleError(ErrorKind.COMPILE, mod.astext(), str(err), 0)
        try:
            ref_outputs = run_gmod(gmod, inputs)
        except Exception as err:
            raise ModuleError(ErrorKind.RUN, mod.astext(), str(err), 0)

        # Build and run modules with different levels of optimization
        for opt_level in range(1, 4):
            try:
                gmod = build_mod(mod, opt_level, params=params)
            except Exception as err:
                raise ModuleError(ErrorKind.COMPILE, mod.astext(), str(err), opt_level)
            try:
                outputs = run_gmod(gmod, inputs)
            except Exception as err:
                raise ModuleError(ErrorKind.RUN, mod.astext(), str(err), opt_level)
            for i, (o, ro) in enumerate(zip(outputs, ref_outputs)):
                if not np.allclose(o, ro, rtol=1e-2, atol=1e-3, equal_nan=True):
                    msg = f'Computation error in output tensor {i}:\n' \
                          f'Expect:\n' \
                          f'{np.array_repr(ro)}\n' \
                          f'Actual:\n' \
                          f'{np.array_repr(o)}'
                    raise ModuleError(ErrorKind.COMPUTE, mod.astext(), msg, opt_level)


def gen_tensor_value(var: relay.Var, rng: Generator):
    var_ty = var.checked_type
    return rng.standard_normal(
        size=[int(d) for d in var_ty.shape], dtype='float64'
    ).astype(var_ty.dtype)


def gen_tensor_value_dict(params: List[relay.Var], rng: Generator):
    return {var.name_hint: gen_tensor_value(var, rng) for var in params}


def build_mod(mod: IRModule, opt_level: int, params: Optional[Dict[str, np.ndarray]] = None):
    with transform.PassContext(opt_level=opt_level, disabled_pass=['AlterOpLayout']):
        lib = relay.build(mod, target='llvm', params=params)
    return GraphModule(lib['default'](cpu()))


def run_gmod(gmod: GraphModule, inputs: Dict[str, np.ndarray]):
    gmod.run(**inputs)
    return [gmod.get_output(i).numpy() for i in range(gmod.get_num_outputs())]
