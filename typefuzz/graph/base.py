from enum import IntEnum, auto
from typing import List, TypeVar, Generic, Dict, Callable, Any, Optional, Tuple

from ..expr.ty import ValueType
from ..solve import TensorType
from ..spec import Op
from ..util import Ref


class VertexKind(IntEnum):
    IN = auto()
    OUT = auto()
    OP = auto()


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


class Operation(Vertex):
    """
    Call of operator on tensor values.
    """
    kind = VertexKind.OP

    def __init__(self, op: Op, ins: List['Value'], outs: List['Value'],
                 attrs: List[Tuple[str, ValueType]]):
        self.op_ = op
        self.ins_ = ins
        self.outs_ = outs
        for out in self.outs_:
            out.def_ = self
        self.attrs_ = attrs


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

    def __init__(self, ins: List[Input], outs: List[Output], ops: List[Operation]):
        self.ins_ = ins
        self.outs_ = outs
        self.ops_ = ops


R = TypeVar('R')


class GraphVisitor(Generic[R]):
    def __init__(self):
        self._methods: Dict[VertexKind, Callable[[Any], R]] = {
            VertexKind.IN: self.visit_input,
            VertexKind.OUT: self.visit_output,
            VertexKind.OP: self.visit_operation,
        }
        self._vert_memo: Dict[Ref[Vertex], R] = {}

    def visit(self, v: Vertex):
        ref = Ref(v)
        if ref in self._vert_memo:
            return self._vert_memo[ref]
        r = self._methods[v.kind](v)
        self._vert_memo[ref] = r
        return r

    def visit_input(self, i: Input) -> R:
        pass

    def visit_output(self, o: Output) -> R:
        pass

    def visit_operation(self, opr: Operation) -> R:
        pass

    def visit_value(self, v: Value):
        pass