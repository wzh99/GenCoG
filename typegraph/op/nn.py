from ..expr import *
from ..spec import Attr, ConstraintSpec, Op


def _conv_dim(data: Expr, weight: Expr, dilation: Expr, pad_b: Expr, pad_e: Expr, stride: Expr):
    dilated_kernel_size = (weight - 1) * dilation + 1
    return (data + pad_b + pad_e - dilated_kernel_size) // stride + 1


_conv2d = ConstraintSpec(
    attrs=[
        Attr('kernel_size', [Var(INT), Var(INT)]),
        Attr('channels', Var(INT)),
        Attr('strides', [Var(INT), Var(INT)]),
        Attr('padding', List(4, lambda _: Var(INT, tmpl=True))),
        Attr('dilation', [Var(INT), Var(INT)]),
        Attr('groups', Var(INT)),
    ],
    in_num=2,
    in_ranks=[4, 4],
    in_dtypes=[Var(), IN[0].dtype],
    in_shapes=[
        [Var(), Var(), Var(), Var()],  # NCHW
        Concat([a('channels'), IN[0].shape[1] // a('groups')], a('kernel_size'))  # OIHW
    ],
    extra=[
        a('channels') % a('groups') == 0,
        IN[0].shape[1] % a('groups') == 0,
    ],
    out_num=1,
    out_ranks=[4],
    out_dtypes=[IN[0].dtype],
    out_shapes=[
        [
            IN[0].shape[0],  # N
            a('channels'),  # C
            _conv_dim(IN[0].shape[2], IN[1].shape[2], a('dilation')[0], a('padding')[0],
                      a('padding')[2], a('strides')[0]),  # H
            _conv_dim(IN[0].shape[3], IN[1].shape[3], a('dilation')[1], a('padding')[1],
                      a('padding')[3], a('strides')[1])  # W
        ]
    ]
)

Op('nn.conv2d', _conv2d)
