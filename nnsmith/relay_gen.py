from numpy.random import Generator
from tvm.relay.frontend import from_onnx
from tvm.relay.transform import InferType

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
]


def nnsmith_gen_relay(opset: List[Type[AbsOpBase]], max_nodes: int, rng: Generator):
    # Generate a random ONNX model
    seed = rng.integers(2 ** 32)
    model_cls = Model.init('onnx')
    model_cls.add_seed_setter()

    # GENERATION
    gen = model_gen(
        opset=opset,
        method='symbolic-cinit',
        seed=seed,
        max_elem_per_tensor=65536,
        max_nodes=max_nodes,
        concr_ph_dim_rng=(1, 4),
    )

    # MATERIALIZATION
    ir = gen.make_concrete()
    model = model_cls.from_gir(ir).native_model
    mod, params = from_onnx(model)
    mod = InferType()(mod)

    return mod, params
