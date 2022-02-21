if __name__ == '__main__':
    from typefuzz.graph.base import Input, Output, Operation, Graph, Value
    from typefuzz.graph.relay import RelayPrinter
    from typefuzz.spec import OpRegistry
    from typefuzz.solve import TensorType
    from typefuzz.expr import DataType
    from tvm import parser

    x = Input(TensorType([32], DataType.f(32)), False)
    y1 = Value(TensorType([16], DataType.f(32)))
    y2 = Value(TensorType([16], DataType.f(32)))
    split = Operation(OpRegistry.get('split'), [('axis', 0), ('indices_or_sections', 2)],
                      [x.value_], [y1, y2])
    z = Value(TensorType([16], DataType.f(32)))
    add = Operation(OpRegistry.get('add'), [], [y1, y2], [z])
    g = Graph([x], [Output(z)], [split, add])
    src = RelayPrinter().print(g)
    print(src)
    mod = parser.parse(src)
    print(mod)
    pass
