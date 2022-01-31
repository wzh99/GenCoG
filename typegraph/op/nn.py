from ..expr import *
from ..spec import Attr, ConstraintSpec, Op


def _conv_dim(data: Expr, weight: Expr, dilation: Expr, pad_b: Expr, pad_e: Expr, stride: Expr):
    dilated_kernel_size = (weight - 1) * dilation + 1
    return (data + pad_b + pad_e - dilated_kernel_size) // stride + 1


def _create_conv2d():
    out_h = _conv_dim(IN[0].shape[2], IN[1].shape[2], a('dilation')[0], a('padding')[0],
                      a('padding')[2], a('strides')[0])
    out_w = _conv_dim(IN[0].shape[3], IN[1].shape[3], a('dilation')[1], a('padding')[1],
                      a('padding')[3], a('strides')[1])
    return ConstraintSpec(
        attrs=[
            Attr('kernel_size', List(2, lambda _: Var(INT, ran=Range(begin=1, end=8), tmpl=True))),
            Attr('channels', Var(INT, ran=Range(begin=1, end=129))),
            Attr('strides', List(2, lambda _: Var(INT, ran=Range(begin=1, end=3), tmpl=True))),
            Attr('padding', List(4, lambda _: Var(INT, tmpl=True))),
            Attr('dilation', List(2, lambda _: Var(INT, ran=Range(end=3), tmpl=True))),
            Attr('groups', Var(INT)),
        ],
        in_num=2,
        in_ranks=[4, 4],
        in_dtypes=List(2, lambda _: Var()),
        in_shapes=[
            [Var(), Var(), Var(), Var()],  # NCHW
            Concat([a('channels'), IN[0].shape[1] // a('groups')], a('kernel_size'))  # OIHW
        ],
        extra=[
            a('channels') % a('groups') == 0,
            IN[0].shape[1] % a('groups') == 0,
            out_h > 0,
            out_w > 0,
        ],
        out_num=1,
        out_ranks=[4],
        out_dtypes=[IN[0].dtype],
        out_shapes=[
            [IN[0].shape[0], a('channels'), out_h, out_w]
        ]
    )


Op('nn.conv2d', _create_conv2d())
