from ..config import config
from ..expr import *
from ..spec import Attr, ConstraintSpec, Op, dim_ran

kernel_ran = iran(1, config['op.max_kernel'])
stride_ran = iran(1, config['op.max_stride'])
pad_ran = iran(0, config['op.max_padding'])
dil_ran = iran(1, config['op.max_dilation'])


def _conv_dim_no_stride(i: Expr, w: Expr, dil: Expr, pad_b: Expr, pad_e: Expr):
    return i + pad_b + pad_e - (w - 1) * dil - 1


def _conv_dim(i: Expr, w: Expr, dil: Expr, pad_b: Expr, pad_e: Expr, stride: Expr):
    return _conv_dim_no_stride(i, w, dil, pad_b, pad_e) // stride + 1


def _create_conv_nd(n: int):
    # Layout
    dim_str = 'DHW'[-n:]
    data_chan_first = 'NC' + dim_str
    data_chan_last = 'N' + dim_str + 'C'
    data_layout_choices = [data_chan_first, data_chan_last]
    kernel_chan_first = 'OI' + dim_str
    kernel_chan_last = dim_str + 'IO'
    kernel_layout_choices = [kernel_chan_first, kernel_chan_last]

    # Dimension
    in_chan = IN[0].shape[LayoutIndex(a('data_layout'), 'C')]
    dims_extra = [
        _conv_dim_no_stride(
            IN[0].shape[LayoutIndex(a('data_layout'), dim_str[i])],
            IN[1].shape[LayoutIndex(a('kernel_layout'), dim_str[i])],
            a('dilation')[i], a('padding')[i], a('padding')[n + i]
        ) >= 0 for i in range(n)
    ]
    out_dims = [
        _conv_dim(
            IN[0].shape[LayoutIndex(a('data_layout'), dim_str[i])],
            IN[1].shape[LayoutIndex(a('kernel_layout'), dim_str[i])],
            a('dilation')[i], a('padding')[i], a('padding')[n + i], a('strides')[i]
        ) for i in range(n)
    ]

    return ConstraintSpec(
        attrs=[
            Attr('kernel_size', List(n, lambda _: Var(INT, ran=kernel_ran, tmpl=True))),
            Attr('channels', Var(INT, ran=dim_ran)),
            Attr('strides', List(n, lambda _: Var(INT, ran=stride_ran, tmpl=True))),
            Attr('padding', List(2 * n, lambda _: Var(INT, ran=pad_ran, tmpl=True))),
            Attr('dilation', List(n, lambda _: Var(INT, ran=dil_ran, tmpl=True))),
            Attr('data_layout', Var(STR)),
            Attr('groups', Var(INT, ran=iran(1, in_chan))),
            Attr('kernel_layout', Var(STR)),
            Attr('out_layout', Var(STR)),
        ],
        in_num=2,
        in_ranks=[n + 2] * 2,
        in_dtypes=List(2, lambda _: Var()),
        in_shapes=[
            LayoutMap(a('data_layout'), data_chan_first, List(n + 2, lambda _: Var(tmpl=True))),
            LayoutMap(
                a('kernel_layout'), kernel_chan_first,
                Concat([a('channels'), in_chan // a('groups')], a('kernel_size'))
            ),
        ],
        extra=
        [
            a('channels') % a('groups') == 0,
            in_chan % a('groups') == 0,
            InSet(a('data_layout'), data_layout_choices),
            InSet(a('kernel_layout'), kernel_layout_choices),
            InSet(a('out_layout'), data_layout_choices),
        ]
        + dims_extra,
        out_num=1,
        out_ranks=[n + 2],
        out_dtypes=[IN[0].dtype],
        out_shapes=[
            LayoutMap(a('out_layout'), data_chan_first, [IN[0].shape[0], a('channels')] + out_dims)
        ]
    )


def _create_conv1d():
    spec = _create_conv_nd(1)
    spec.reset_attr(Attr('groups', 1))  # grouped 1D convolution is not supported
    return spec


Op('nn.conv1d', _create_conv1d())
Op('nn.conv2d', _create_conv_nd(2))
Op('nn.conv3d', _create_conv_nd(3))


def _conv_trans_dim(i: Expr, w: Expr, stride: Expr, dil: Expr, pad_b: Expr, pad_e: Expr,
                    out_pad: Expr):
    return (i - 1) * stride + (w - 1) * dil - pad_b - pad_e + out_pad + 1


def _create_conv_trans_nd(n: int):
    # Layout
    dim_str = 'DHW'[-n:]
    data_chan_first = 'NC' + dim_str
    data_chan_last = 'N' + dim_str + 'C'
    data_layout_choices = [data_chan_first, data_chan_last]
    kernel_chan_first = 'OI' + dim_str
    kernel_chan_last = dim_str + 'IO'
    kernel_layout_choices = [kernel_chan_first, kernel_chan_last]

    # Dimension
    in_chan = IN[0].shape[LayoutIndex(a('data_layout'), 'C')]
    out_dims = [
        _conv_trans_dim(
            IN[0].shape[LayoutIndex(a('data_layout'), dim_str[i])],
            IN[1].shape[LayoutIndex(a('kernel_layout'), dim_str[i])],
            a('strides')[i], a('dilation')[i], a('padding')[i], a('padding')[n + i],
            a('output_padding')[i]
        ) for i in range(n)
    ]
    dims_extra = [d >= 1 for d in out_dims]

    return ConstraintSpec(
        attrs=[
            Attr('kernel_size', List(n, lambda _: Var(INT, ran=kernel_ran, tmpl=True))),
            Attr('channels', Var(INT, ran=dim_ran)),
            Attr('strides', List(n, lambda _: Var(INT, ran=stride_ran, tmpl=True))),
            Attr('padding', List(2 * n, lambda _: Var(INT, ran=pad_ran, tmpl=True))),
            Attr('output_padding', List(n, lambda _: Var(INT, ran=pad_ran, tmpl=True))),
            Attr('dilation', List(n, lambda _: Var(INT, ran=dil_ran, tmpl=True))),
            Attr('data_layout', Var(STR)),
            Attr('groups', Var(INT, ran=iran(1, in_chan))),
            Attr('kernel_layout', Var(STR)),
            Attr('out_layout', Var(STR)),
        ],
        in_num=2,
        in_ranks=[n + 2] * 2,
        in_dtypes=List(2, lambda _: Var()),
        in_shapes=[
            LayoutMap(a('data_layout'), data_chan_first, List(n + 2, lambda _: Var(tmpl=True))),
            LayoutMap(
                a('kernel_layout'), kernel_chan_first,
                Concat([a('channels') // a('groups'), in_chan], a('kernel_size'))
            ),
        ],
        extra=
        [
            a('channels') % a('groups') == 0,
            in_chan % a('groups') == 0,
            InSet(a('data_layout'), data_layout_choices),
            InSet(a('kernel_layout'), kernel_layout_choices),
            InSet(a('out_layout'), data_layout_choices),
        ]
        + dims_extra,
        out_num=1,
        out_ranks=[n + 2],
        out_dtypes=[IN[0].dtype],
        out_shapes=[
            LayoutMap(a('out_layout'), data_chan_first, [IN[0].shape[0], a('channels')] + out_dims)
        ]
    )


Op('nn.conv1d_transpose', _create_conv_trans_nd(1))
Op('nn.conv2d_transpose', _create_conv_trans_nd(2))
Op('nn.conv3d_transpose', _create_conv_trans_nd(3))
