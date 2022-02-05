from ..config import config
from ..expr import *
from ..spec import Attr, ConstraintSpec, Op, dim_ran

kernel_ran = iran(1, config['op.max_kernel'])
stride_ran = iran(1, config['op.max_stride'])
pad_ran = iran(0, config['op.max_padding'])
dil_ran = iran(1, config['op.max_dilation'])


def _conv_dim_no_stride(data: Expr, weight: Expr, dilation: Expr, pad_b: Expr, pad_e: Expr):
    return data + pad_b + pad_e - (weight - 1) * dilation - 1


def _conv_dim(data: Expr, weight: Expr, dilation: Expr, pad_b: Expr, pad_e: Expr, stride: Expr):
    return _conv_dim_no_stride(data, weight, dilation, pad_b, pad_e) // stride + 1


def _create_conv2d():
    out_h_ns = _conv_dim_no_stride(IN[0].shape[2], IN[1].shape[2], a('dilation')[0],
                                   a('padding')[0], a('padding')[2])
    out_w_ns = _conv_dim_no_stride(IN[0].shape[3], IN[1].shape[3], a('dilation')[1],
                                   a('padding')[1], a('padding')[3])
    out_h = _conv_dim(IN[0].shape[2], IN[1].shape[2], a('dilation')[0], a('padding')[0],
                      a('padding')[2], a('strides')[0])
    out_w = _conv_dim(IN[0].shape[3], IN[1].shape[3], a('dilation')[1], a('padding')[1],
                      a('padding')[3], a('strides')[1])
    return ConstraintSpec(
        attrs=[
            Attr('kernel_size', List(2, lambda _: Var(INT, ran=kernel_ran, tmpl=True))),
            Attr('channels', Var(INT, ran=dim_ran)),
            Attr('strides', List(2, lambda _: Var(INT, ran=stride_ran, tmpl=True))),
            Attr('padding', List(4, lambda _: Var(INT, ran=pad_ran, tmpl=True))),
            Attr('dilation', List(2, lambda _: Var(INT, ran=dil_ran, tmpl=True))),
            Attr('groups', Var(INT, ran=iran(1, IN[0].shape[1]))),
        ],
        in_num=2,
        in_ranks=[4, 4],
        in_dtypes=List(2, lambda _: Var()),
        in_shapes=[
            List(4, lambda _: Var(tmpl=True)),  # NCHW
            Concat([a('channels'), IN[0].shape[1] // a('groups')], a('kernel_size'))  # OIHW
        ],
        extra=[
            a('channels') % a('groups') == 0,
            IN[0].shape[1] % a('groups') == 0,
            out_h_ns >= 0,
            out_w_ns >= 0,
        ],
        out_num=1,
        out_ranks=[4],
        out_dtypes=[IN[0].dtype],
        out_shapes=[
            [IN[0].shape[0], a('channels'), out_h, out_w]
        ]
    )


Op('nn.conv2d', _create_conv2d())
