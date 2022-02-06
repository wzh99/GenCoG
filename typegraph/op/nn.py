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


def _create_conv_nd(n: int):
    dims_extra = [
        _conv_dim_no_stride(IN[0].shape[2 + i], IN[1].shape[2 + i], a('dilation')[i],
                            a('padding')[i], a('padding')[n + i]) >= 0
        for i in range(n)
    ]
    out_dims = [
        _conv_dim(IN[0].shape[2 + i], IN[1].shape[2 + i], a('dilation')[i],
                  a('padding')[i], a('padding')[n + i], a('strides')[i])
        for i in range(n)
    ]
    return ConstraintSpec(
        attrs=[
            Attr('kernel_size', List(n, lambda _: Var(INT, ran=kernel_ran, tmpl=True))),
            Attr('channels', Var(INT, ran=dim_ran)),
            Attr('strides', List(n, lambda _: Var(INT, ran=stride_ran, tmpl=True))),
            Attr('padding', List(2 * n, lambda _: Var(INT, ran=pad_ran, tmpl=True))),
            Attr('dilation', List(n, lambda _: Var(INT, ran=dil_ran, tmpl=True))),
            Attr('groups', Var(INT, ran=iran(1, IN[0].shape[1]))),
        ],
        in_num=2,
        in_ranks=[n + 2] * 2,
        in_dtypes=List(2, lambda _: Var()),
        in_shapes=[
            List(n + 2, lambda _: Var(tmpl=True)),  # NCHW
            Concat([a('channels'), IN[0].shape[1] // a('groups')], a('kernel_size'))  # OIHW
        ],
        extra=[a('channels') % a('groups') == 0, IN[0].shape[1] % a('groups') == 0] + dims_extra,
        out_num=1,
        out_ranks=[n + 2],
        out_dtypes=[IN[0].dtype],
        out_shapes=[
            [IN[0].shape[0], a('channels')] + out_dims
        ]
    )


def _create_conv1d():
    spec = _create_conv_nd(1)
    spec.reset_attr(Attr('groups', 1))  # grouped 1D convolution is not supported
    return spec


Op('nn.conv1d', _create_conv1d())
Op('nn.conv2d', _create_conv_nd(2))
Op('nn.conv3d', _create_conv_nd(3))
