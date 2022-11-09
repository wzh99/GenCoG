from typing import Dict, List, cast

import numpy as np
from tvm import relay, tir, runtime, ir
from tvm.ir import IRModule

from .base import GraphVisitor, Value, Graph, VertexKind, Input, Output, Operation, TensorType
from ..expr.ty import ValueType, DataType
from ..spec import OpRegistry
from ..util import NameGenerator, CodeBuffer

# Operators that accept tuple as input
tuple_in_ops = {
    'concatenate',
}

# Operators that return a tuple, no matter how many output values they produce
tuple_out_ops = {
    'split',
}


def print_relay(g: Graph):
    return RelayPrinter().print(g)


def fmt_val(v: ValueType):
    if isinstance(v, (bool, int, DataType)):
        return str(v)
    elif isinstance(v, float):
        return str(v) + 'f'
    elif isinstance(v, str):
        return '"' + v + '"'
    elif isinstance(v, (tuple, list)):
        return '[' + ', '.join(fmt_val(e) for e in v) + ']'
    elif v is None:
        return fmt_val([])
    else:
        assert False


class RelayPrinter(GraphVisitor[None]):
    def __init__(self):
        super().__init__()
        self._buf = CodeBuffer()
        self._val_names: Dict[Value, str] = {}
        self._arg_gen = NameGenerator('%x')
        self._res_gen = NameGenerator('%')

    def print(self, g: Graph):
        # Function signature
        self._buf.writeln('#[version = "0.0.5"]')
        self._buf.write('def @main')
        self._buf.write_pos(
            map(lambda i: lambda: self._buf.write(
                f'{self.visit_value(i.value_)}: {i.value_.type_}'), g.inputs_)
        )
        self._buf.write(' -> ')
        self._buf.write_pos(
            map(lambda o: lambda: self._buf.write(str(o.value_.type_)), g.outputs_)
        )

        # Function body
        self._buf.writeln(' {')
        with self._buf.indent():
            for out in g.outputs_:
                self.visit(out)
            out_str = str(tuple(self.visit_value(o.value_) for o in g.outputs_)).replace('\'', '')
            self._buf.writeln(out_str)
        self._buf.writeln('}')

        return str(self._buf)

    def visit_input(self, i: Input):
        return

    def visit_output(self, o: Output):
        return self.visit(o.value_.def_)

    def visit_operation(self, opr: Operation):
        # Visit predecessors
        for v in opr.inputs_:
            self.visit(v.def_)

        # Print output value
        op_name = opr.op_.name_
        tup_out = len(opr.outputs_) > 1 or op_name in tuple_out_ops
        if tup_out:
            tup_name = self._res_gen.generate()
            self._buf.write(tup_name)
        else:
            tup_name = ''
            self._buf.write(self.visit_value(opr.outputs_[0]))
        self._buf.write(' = ')

        # Print operator call
        self._buf.write(opr.op_.name_)
        args = map(lambda v: self.visit_value(v), opr.inputs_)
        if op_name in tuple_in_ops:
            arg_str = str(tuple(args)).replace('\'', '')
        else:
            arg_str = ', '.join(args)
        self._buf.write_pos([
            lambda: self._buf.write(arg_str),
            lambda: self._buf.write_named(
                map(lambda a: (a[0], lambda: self._buf.write(fmt_val(a[1]))), opr.attrs_),
                prefix='', suffix=''
            )
        ])
        ty_str = repr(tuple(out.type_ for out in opr.outputs_)) if tup_out else repr(
            opr.outputs_[0].type_)
        self._buf.writeln(f'; /* ty={ty_str} */')

        # Unpack tuple
        if tup_out:
            for i, v in enumerate(opr.outputs_):
                self._buf.writeln(f'{self.visit_value(v)} = {tup_name}.{i};')

    def visit_value(self, v: Value):
        if v in self._val_names:
            return self._val_names[v]
        if v.def_.kind == VertexKind.IN:
            name = self._arg_gen.generate()
        else:
            name = self._res_gen.generate()
        self._val_names[v] = name
        return name


def build_graph(mod: IRModule, params: Dict[str, np.ndarray]):
    return GraphBuilder(params).visit(mod['main'])


class GraphBuilder(relay.ExprFunctor):
    def __init__(self, params: Dict[str, np.ndarray]):
        super().__init__()
        self._params = params
        self._name2val: Dict[str, Value] = {}
        self._inputs: List[Input] = []
        self._oprs: List[Operation] = []

    def visit_function(self, fn: relay.Function):
        # Create inputs
        self._inputs = [Input(_cvt_type(var.checked_type), var.name_hint in self._params)
                        for var in fn.params]
        self._name2val = {p.name_hint: inp.value_ for p, inp in zip(fn.params, self._inputs)}

        # Build operations
        if isinstance(fn.body, (relay.Call, relay.TupleGetItem, relay.Var)):
            outputs = [Output(self.visit(fn.body))]
        elif isinstance(fn.body, relay.Tuple):
            outputs = [Output(self.visit(f)) for f in fn.body.fields]
        else:
            raise TypeError('{} not supported.'.format(type(fn.body).__name__))

        # Create graph
        return Graph(self._inputs, outputs, self._oprs)

    def visit_var(self, var: relay.Var):
        return self._name2val[var.name_hint]

    def visit_constant(self, const: relay.Constant):
        inp = Input(_cvt_type(const.checked_type), True)
        self._inputs.append(inp)
        return inp.value_

    def visit_tuple(self, tup: relay.Tuple):
        return [self.visit(f) for f in tup.fields]

    def visit_tuple_getitem(self, getitem: relay.TupleGetItem):
        opr = cast(Value, self.visit(getitem.tuple_value)).def_
        return cast(Operation, opr).outputs_[getitem.index]

    def visit_call(self, call: relay.Call):
        # Collect input values
        name = call.op.name
        if name in tuple_in_ops:
            inputs = self.visit_tuple(call.args[0])
        else:
            inputs = [self.visit(a) for a in call.args]

        # Convert attribute values
        if call.attrs is None or (not hasattr(call.attrs, 'keys')):
            attrs = []
        else:
            attrs = [(str(k), _cvt_ir_value(call.attrs[k])) for k in call.attrs.keys()]

        # Create output values
        out_ty = call.checked_type
        if isinstance(out_ty, relay.TensorType):
            outputs = [Value(_cvt_type(out_ty))]
        elif isinstance(out_ty, relay.TupleType):
            outputs = [Value(_cvt_type(f)) for f in out_ty.fields]
        else:
            raise TypeError('{} not supported.'.format(type(out_ty).__name__))

        # Create operation
        opr = Operation(OpRegistry.get(name), attrs, inputs, outputs)
        self._oprs.append(opr)

        return opr.outputs_[0]

    def visit_let(self, _):
        raise NotImplemented

    def visit_if(self, _):
        raise NotImplemented

    def visit_global_var(self, _):
        raise NotImplemented

    def visit_op(self, _):
        raise NotImplemented

    def visit_ref_create(self, _):
        raise NotImplemented

    def visit_ref_write(self, _):
        raise NotImplemented

    def visit_ref_read(self, _):
        raise NotImplemented

    def visit_constructor(self, _):
        raise NotImplemented

    def visit_match(self, _):
        raise NotImplemented


def _cvt_type(ty: relay.TensorType):
    return TensorType(_cvt_ir_value(ty.shape), DataType.from_str(ty.dtype))


def _cvt_ir_value(val) -> ValueType:
    if isinstance(val, (tir.IntImm, tir.FloatImm, tir.StringImm)):
        return val.value
    elif isinstance(val, runtime.String):
        return str(val)
    elif isinstance(val, (list, ir.Array)):
        return tuple(_cvt_ir_value(e) for e in val)
    else:
        return val
