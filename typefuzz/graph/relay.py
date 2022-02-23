from typing import Dict

from .base import GraphVisitor, Value, Graph, VertexKind, Input, Output, Operation
from ..expr.ty import ValueType, DataType
from ..util import Ref, NameGenerator, CodeBuffer

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
    if isinstance(v, (bool, int, float, DataType)):
        return str(v)
    elif isinstance(v, str):
        return '"' + v + '"'
    elif isinstance(v, (tuple, list)):
        return '[' + ', '.join(fmt_val(e) for e in v) + ']'
    else:
        assert False


class RelayPrinter(GraphVisitor[None]):
    def __init__(self):
        super().__init__()
        self._buf = CodeBuffer()
        self._val_id: Dict[Ref[Value], str] = {}
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
        self._buf.writeln(';')

        # Unpack tuple
        if tup_out:
            for i, v in enumerate(opr.outputs_):
                self._buf.writeln(f'{self.visit_value(v)} = {tup_name}.{i};')

    def visit_value(self, v: Value):
        ref = Ref(v)
        if ref in self._val_id:
            return self._val_id[ref]
        if v.def_.kind == VertexKind.IN:
            name = self._arg_gen.generate()
        else:
            name = self._res_gen.generate()
        self._val_id[ref] = name
        return name
