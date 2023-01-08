from numpy.random import Generator
from tvm.relay.frontend import from_onnx

from .abstract.op import *
from .graph_gen import model_gen
from .materialize import Model
from .materialize.torch.dialect import *

common_opset = [
    Sigmoid,
    Add,
    Sub,
    Mul,
    Max,
    Min,
    Reshape,
    Transpose,
    Concat1,
    Concat2,
    Concat3,
    Concat4,
    Slice,
    ReLU,
    LeakyReLU,
    PReLU,
    Softmax,
    Conv1d,
    NCHWConv2d,
    MaxPool2d,
    AvgPool2d,
    ConstPad,
    BatchNorm2d,
    Linear,
    Flatten,
]


def nnsmith_gen_relay(opset: List[Type[AbsOpBase]], rng: Generator):
    # Generate a random ONNX model
    seed = rng.integers(2 ** 32)
    ModelType = Model.init('onnx')
    ModelType.add_seed_setter()

    # GENERATION
    gen = model_gen(
        opset=opset,
        method='concolic',
        seed=seed,
        max_elem_per_tensor=65536,
        max_nodes=32,
        concr_ph_dim_rng=(1, 4),
    )

    # MATERIALIZATION
    ir = gen.make_concrete()
    model = ModelType.from_gir(ir).native_model
    dtypes = {name: svar.dtype.name for name, svar in ir.vars.items()}
    mod, params = from_onnx(model, dtype=dtypes)

    return mod, params
