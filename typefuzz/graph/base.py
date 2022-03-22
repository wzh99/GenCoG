from enum import IntEnum, auto
from typing import List, TypeVar, Generic, Dict, Callable, Any, Optional, Tuple

from ..expr.ty import ValueType
from ..solve import TensorType
from ..spec import Op


class VertexKind(IntEnum):
    IN = auto()
    OUT = auto()
    OPR = auto()


class Vertex:
    """
    Base class of computation graph vertex.
    """
    kind: VertexKind


class Input(Vertex):
    """
    Input placeholder of computation graph.
    """
    kind = VertexKind.IN

    def __init__(self, ty: TensorType, is_param: bool):
        self.value_ = Value(ty, def_vert=self)
        self.is_param_ = is_param


class Output(Vertex):
    """
    Output indicator of computation graph.
    """
    kind = VertexKind.OUT

    def __init__(self, value: 'Value'):
        self.value_ = value
        value.uses_.append(self)


class Operation(Vertex):
    """
    Call of operator on tensor values.
    """
    kind = VertexKind.OPR

    def __init__(self, op: Op, attrs: List[Tuple[str, ValueType]],
                 inputs: List['Value'], outputs: List['Value']):
        self.op_ = op
        self.attrs_ = attrs
        self.inputs_ = inputs
        for i in self.inputs_:
            i.uses_.append(self)
        self.outputs_ = outputs
        for o in self.outputs_:
            o.def_ = self


class Value:
    """
    Tensor value defined by vertex.
    """

    def __init__(self, ty: TensorType, def_vert: Optional[Vertex] = None):
        self.type_ = ty
        self.def_ = def_vert
        self.uses_: List[Vertex] = []


class Graph:
    """
    Computation graph.
    """

    def __init__(self, ins: List[Input], outs: List[Output], oprs_: List[Operation]):
        self.inputs_ = ins
        self.outputs_ = outs
        self.oprs_ = oprs_


R = TypeVar('R')


class GraphVisitor(Generic[R]):
    def __init__(self):
        self._methods: Dict[VertexKind, Callable[[Any], R]] = {
            VertexKind.IN: self.visit_input,
            VertexKind.OUT: self.visit_output,
            VertexKind.OPR: self.visit_operation,
        }
        self._vert_memo: Dict[Vertex, R] = {}

    def visit(self, v: Vertex):
        if v in self._vert_memo:
            return self._vert_memo[v]
        r = self._methods[v.kind](v)
        self._vert_memo[v] = r
        return r

    def visit_input(self, i: Input) -> R:
        pass

    def visit_output(self, o: Output) -> R:
        pass

    def visit_operation(self, opr: Operation) -> R:
        pass

    def visit_value(self, v: Value):
        pass
