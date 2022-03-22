from typing import Dict

from graphviz import Digraph

from .base import GraphVisitor, Input, Value, Output, Operation, Graph
from .relay import fmt_val
from ..solve import TensorType
from ..util import NameGenerator


def visualize(graph: Graph, name: str, directory: str, fontname: str = 'Linux Biolinum O'):
    GraphVisualizer(name, directory, fontname).visualize(graph)


class GraphVisualizer(GraphVisitor[None]):
    def __init__(self, name: str, directory: str, fontname: str):
        super().__init__()
        self._viz = Digraph(
            name=name,
            directory=directory,
            format='pdf',
            node_attr={
                'shape': 'record',
                'fontname': fontname,
            }
        )
        self._def_str: Dict[Value, str] = dict()
        self._inp_gen = NameGenerator('in')
        self._out_gen = NameGenerator('out')
        self._opr_gen = NameGenerator('opr')

    def visualize(self, graph: Graph):
        for out in graph.outputs_:
            self.visit(out)
        self._viz.render()

    def visit_input(self, i: Input):
        name = self._inp_gen.generate()
        self._viz.node(name, label=f'{name}: {self._fmt_ty(i.value_.type_)}')
        self._def_str[i.value_] = name

    def visit_output(self, o: Output):
        self.visit(o.value_.def_)
        name = self._out_gen.generate()
        self._viz.node(name, label=f'{name}: {self._fmt_ty(o.value_.type_)}')
        self._viz.edge(self._def_str[o.value_], name)

    def visit_operation(self, opr: Operation):
        for v in opr.inputs_:
            self.visit(v.def_)
        name = self._opr_gen.generate()
        attr_label = '\\n'.join(f'{n}={fmt_val(v)}' for n, v in opr.attrs_)
        input_label = '{' + '|'.join(f'<i{i}>{self._fmt_ty(v.type_)}'
                                     for i, v in enumerate(opr.inputs_)) + '}'
        output_label = '{' + '|'.join(f'<o{i}>{self._fmt_ty(v.type_)}'
                                      for i, v in enumerate(opr.outputs_)) + '}'
        opr_label = '{' + f'{input_label}|{opr.op_.name_}\\n' \
                          f'{attr_label}|{output_label}' + '}'
        self._viz.node(name, label=opr_label)
        for i, v in enumerate(opr.inputs_):
            self._viz.edge(self._def_str[v], f'{name}:i{i}')
        for i, v in enumerate(opr.outputs_):
            self._def_str[v] = f'{name}:o{i}'

    @staticmethod
    def _fmt_ty(ty: TensorType):
        return f'[{tuple(ty.shape_)}, {ty.dtype_}]'
