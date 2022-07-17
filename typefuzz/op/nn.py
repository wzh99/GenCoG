from .math import create_identity
from ..config import params
from ..expr import *
from ..expr.ty import float_dtypes
from ..spec import Attr, TypeSpec, Op, dim_ran, max_rank, dl_rank_ran, rank_ran

kernel_ran = iran(1, params['op.max_kernel'])
stride_ran = iran(1, params['op.max_stride'])
pad_ran = iran(0, params['op.max_padding'])
dil_ran = iran(1, params['op.max_dilation'])

Op('nn.relu', create_identity)


def _create_leaky_relu():
    spec = create_identity()
    spec.add_attr(Attr('alpha', Var(FLOAT, ran=Range(0., 1.))))
    return spec


Op('nn.leaky_relu', _create_leaky_relu)


def _create_bcast_axis():
    spec = create_identity()
    spec.add_attr(Attr('axis', 1 if TypeSpec.for_graph else Var(
        INT, ran=Range(end=IN[0].rank))))
    spec.in_num = 2
    spec.in_ranks = [Var(), 1]
    spec.in_dtypes = List(2, lambda _: Var())
    spec.in_shapes = [
        List(IN[0].rank, lambda _: Var(tmpl=True)),
        [IN[0].shape[a('axis')]]
    ]
    return spec


Op('nn.prelu', _create_bcast_axis, params=[1])
Op('nn.bias_add', _create_bcast_axis, params=[1])


def _create_softmax():
    spec = create_identity()
    if TypeSpec.for_graph:
        spec.add_attr(Attr('axis', 1))
        spec.in_ranks = [Var(ran=dl_rank_ran)]
        spec.in_dtypes = [Var(choices=float_dtypes)]
    else:
        spec.add_attr(Attr('axis', Var(INT, ran=Range(end=IN[0].rank))))
    return spec


Op('nn.softmax', _create_softmax)


def _conv_dim_no_stride(i: Expr, w: Expr, dil: Expr, pad_b: Expr, pad_e: Expr):
    return i + pad_b + pad_e - (w - 1) * dil - 1


def _conv_dim(i: Expr, w: Expr, dil: Expr, pad_b: Expr, pad_e: Expr, stride: Expr):
    return _conv_dim_no_stride(i, w, dil, pad_b, pad_e) // stride + 1


def _create_conv_nd(n: int):
    if TypeSpec.for_graph:
        in_chan = IN[0].shape[1]
        out_dims = [
            _conv_dim(
                IN[0].shape[2 + i], a('kernel_size')[i], a('dilation')[i],
                a('padding')[i], a('padding')[n + i], a('strides')[i]
            ) for i in range(n)
        ]
        return TypeSpec(
            attrs=[
                Attr('kernel_size', [Var(INT, ran=kernel_ran) for _ in range(n)]),
                Attr('channels', Var(INT, ran=dim_ran)),
                Attr('strides', [Var(INT, ran=stride_ran) for _ in range(n)]),
                Attr('padding', [Var(INT, ran=pad_ran) for _ in range(2 * n)]),
                Attr('dilation', [Var(INT, ran=dil_ran) for _ in range(n)]),
                Attr('groups', Var(INT, ran=iran(1, in_chan))),
            ],
            in_num=2,
            in_ranks=[n + 2] * 2,
            in_dtypes=List(2, lambda _: Var()),
            in_shapes=[
                [Var() for _ in range(n + 2)],
                Concat([a('channels'), in_chan // a('groups')], a('kernel_size')),
            ],
            extra=
            [
                a('channels') % a('groups') == 0,
                in_chan % a('groups') == 0,
            ] + [
                _conv_dim_no_stride(
                    IN[0].shape[2 + i], a('kernel_size')[i], a('dilation')[i],
                    a('padding')[i], a('padding')[n + i]
                ) >= 0 for i in range(n)
            ] + [d == IN[0].shape[2 + i] for i, d in enumerate(out_dims)],
            out_num=1,
            out_ranks=[n + 2],
            out_dtypes=[IN[0].dtype],
            out_shapes=[
                [IN[0].shape[0], a('channels')] + out_dims
            ]
        )

    # Layout
    dim_str = 'DHW'[-n:]
    data_chan_first = 'NC' + dim_str
    data_chan_last = 'N' + dim_str + 'C'
    data_layout_choices = [data_chan_first, data_chan_last]
    kernel_chan_first = 'OI' + dim_str
    kernel_chan_last = dim_str + 'IO'
    in_chan = IN[0].shape[LayoutIndex(a('data_layout'), 'C')]
    is_dw = And(a('groups') == a('channels'), a('groups') == in_chan)
    kernel_layout = Cond(a('data_layout') == data_chan_first, kernel_chan_first,
                         Cond(is_dw, dim_str + 'OI', kernel_chan_last))

    # Dimension
    dims_extra = [
        _conv_dim_no_stride(
            IN[0].shape[LayoutIndex(a('data_layout'), dim_str[i])],
            a('kernel_size')[i], a('dilation')[i], a('padding')[i], a('padding')[n + i]
        ) >= 0 for i in range(n)
    ]
    out_dims = [
        _conv_dim(
            IN[0].shape[LayoutIndex(a('data_layout'), dim_str[i])], a('kernel_size')[i],
            a('dilation')[i], a('padding')[i], a('padding')[n + i], a('strides')[i]
        ) for i in range(n)
    ]

    return TypeSpec(
        attrs=[
            Attr('kernel_size', [Var(INT, ran=kernel_ran) for _ in range(n)]),
            Attr('channels', Var(INT, ran=dim_ran)),
            Attr('strides', [Var(INT, ran=stride_ran) for _ in range(n)]),
            Attr('padding', [Var(INT, ran=pad_ran) for _ in range(2 * n)]),
            Attr('dilation', [Var(INT, ran=dil_ran) for _ in range(n)]),
            Attr('data_layout', Var(STR, choices=data_layout_choices)),
            Attr('groups', Var(INT, ran=iran(1, in_chan))),
            Attr('kernel_layout', kernel_layout),
            Attr('out_layout', Var(STR, choices=data_layout_choices)),
        ],
        in_num=2,
        in_ranks=[n + 2] * 2,
        in_dtypes=List(2, lambda _: Var()),
        in_shapes=[
            LayoutMap(a('data_layout'), data_chan_first, [Var() for _ in range(n + 2)]),
            LayoutMap(
                a('kernel_layout'), kernel_chan_first,
                Concat([a('channels'), in_chan // a('groups')], a('kernel_size'))
            ),
        ],
        extra=
        [
            a('channels') % a('groups') == 0,
            in_chan % a('groups') == 0,
        ] + dims_extra,
        out_num=1,
        out_ranks=[n + 2],
        out_dtypes=[IN[0].dtype],
        out_shapes=[
            LayoutMap(a('out_layout'), data_chan_first, [IN[0].shape[0], a('channels')] + out_dims)
        ]
    )


def _create_conv_nd_no_group(n: int):
    if TypeSpec.for_graph:
        in_chan = IN[0].shape[1]
        out_dims = [
            _conv_dim(
                IN[0].shape[2 + i], a('kernel_size')[i], a('dilation')[i],
                a('padding')[i], a('padding')[n + i], a('strides')[i]
            ) for i in range(n)
        ]
        return TypeSpec(
            attrs=[
                Attr('kernel_size', [Var(INT, ran=kernel_ran) for _ in range(n)]),
                Attr('channels', Var(INT, ran=dim_ran)),
                Attr('strides', [Var(INT, ran=stride_ran) for _ in range(n)]),
                Attr('padding', [Var(INT, ran=pad_ran) for _ in range(2 * n)]),
                Attr('dilation', [Var(INT, ran=dil_ran) for _ in range(n)]),
            ],
            in_num=2,
            in_ranks=[n + 2] * 2,
            in_dtypes=List(2, lambda _: Var()),
            in_shapes=[
                [Var() for _ in range(n + 2)],
                Concat([a('channels'), in_chan], a('kernel_size')),
            ],
            extra=
            [
                _conv_dim_no_stride(
                    IN[0].shape[2 + i], a('kernel_size')[i], a('dilation')[i],
                    a('padding')[i], a('padding')[n + i]
                ) >= 0 for i in range(n)
            ] + [d == IN[0].shape[2 + i] for i, d in enumerate(out_dims)],
            out_num=1,
            out_ranks=[n + 2],
            out_dtypes=[IN[0].dtype],
            out_shapes=[
                [IN[0].shape[0], a('channels')] + out_dims
            ]
        )

    # Layout
    dim_str = 'DHW'[-n:]
    data_chan_first = 'NC' + dim_str
    data_chan_last = 'N' + dim_str + 'C'
    data_layout_choices = [data_chan_first, data_chan_last]
    kernel_chan_first = 'OI' + dim_str
    kernel_chan_last = dim_str + 'IO'
    in_chan = IN[0].shape[LayoutIndex(a('data_layout'), 'C')]

    # Dimension
    dims_extra = [
        _conv_dim_no_stride(
            IN[0].shape[LayoutIndex(a('data_layout'), dim_str[i])],
            a('kernel_size')[i], a('dilation')[i], a('padding')[i], a('padding')[n + i]
        ) >= 0 for i in range(n)
    ]
    out_dims = [
        _conv_dim(
            IN[0].shape[LayoutIndex(a('data_layout'), dim_str[i])], a('kernel_size')[i],
            a('dilation')[i], a('padding')[i], a('padding')[n + i], a('strides')[i]
        ) for i in range(n)
    ]

    return TypeSpec(
        attrs=[
            Attr('kernel_size', [Var(INT, ran=kernel_ran) for _ in range(n)]),
            Attr('channels', Var(INT, ran=dim_ran)),
            Attr('strides', [Var(INT, ran=stride_ran) for _ in range(n)]),
            Attr('padding', [Var(INT, ran=pad_ran) for _ in range(2 * n)]),
            Attr('dilation', [Var(INT, ran=dil_ran) for _ in range(n)]),
            Attr('data_layout', Var(STR, choices=data_layout_choices)),
            Attr('kernel_layout', Cond(a('data_layout') == data_chan_first, kernel_chan_first,
                                       kernel_chan_last)),
        ],
        in_num=2,
        in_ranks=[n + 2] * 2,
        in_dtypes=List(2, lambda _: Var()),
        in_shapes=[
            LayoutMap(a('data_layout'), data_chan_first, [Var() for _ in range(n + 2)]),
            LayoutMap(
                a('kernel_layout'), kernel_chan_first,
                Concat([a('channels'), in_chan], a('kernel_size'))
            ),
        ],
        extra=dims_extra,
        out_num=1,
        out_ranks=[n + 2],
        out_dtypes=[IN[0].dtype],
        out_shapes=[
            LayoutMap(a('data_layout'), data_chan_first, [IN[0].shape[0], a('channels')] + out_dims)
        ]
    )


Op('nn.conv1d', lambda: _create_conv_nd_no_group(1), params=[1])
Op('nn.conv2d', lambda: _create_conv_nd(2), params=[1])
Op('nn.conv3d', lambda: _create_conv_nd(3), params=[1])


def _conv_trans_dim(i: Expr, w: Expr, stride: Expr, dil: Expr, pad_b: Expr, pad_e: Expr,
                    out_pad: Expr):
    return (i - 1) * stride + (w - 1) * dil - pad_b - pad_e + out_pad + 1


def _create_conv_trans_nd(n: int):
    if TypeSpec.for_graph:
        in_chan = IN[0].shape[1]
        out_dims = [
            _conv_trans_dim(
                IN[0].shape[2 + i], a('kernel_size')[i], a('strides')[i], a('dilation')[i],
                a('padding')[i], a('padding')[n + i], a('output_padding')[i]
            ) for i in range(n)
        ]

        return TypeSpec(
            attrs=[
                Attr('kernel_size', [Var(INT, ran=kernel_ran) for _ in range(n)]),
                Attr('channels', Var(INT, ran=dim_ran)),
                Attr('strides', [Var(INT, ran=stride_ran) for _ in range(n)]),
                Attr('padding', [Var(INT, ran=pad_ran) for _ in range(2 * n)]),
                Attr('output_padding',
                     [Var(INT, ran=Range(end=a('strides')[i])) for i in range(n)]),
                Attr('dilation', [1] * n),
                Attr('groups', Var(INT, ran=iran(1, in_chan))),
            ],
            in_num=2,
            in_ranks=[n + 2] * 2,
            in_dtypes=List(2, lambda _: Var()),
            in_shapes=[
                [Var() for _ in range(n + 2)],
                Concat([a('channels') // a('groups'), in_chan], a('kernel_size')),
            ],
            extra=
            [
                a('channels') % a('groups') == 0,
                in_chan % a('groups') == 0,
            ] + [d == IN[0].shape[2 + i] for i, d in enumerate(out_dims)],
            out_num=1,
            out_ranks=[n + 2],
            out_dtypes=[IN[0].dtype],
            out_shapes=[
                [IN[0].shape[0], a('channels')] + out_dims
            ]
        )

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
            IN[0].shape[LayoutIndex(a('data_layout'), dim_str[i])], a('kernel_size')[i],
            a('strides')[i], a('dilation')[i], a('padding')[i], a('padding')[n + i],
            a('output_padding')[i]
        ) for i in range(n)
    ]
    dims_extra = [d >= 1 for d in out_dims]

    return TypeSpec(
        attrs=[
            Attr('kernel_size', [Var(INT, ran=kernel_ran) for _ in range(n)]),
            Attr('channels', Var(INT, ran=dim_ran)),
            Attr('strides', [Var(INT, ran=stride_ran) for _ in range(n)]),
            Attr('padding', [Var(INT, ran=pad_ran) for _ in range(2 * n)]),
            Attr('output_padding', [Var(INT, ran=Range(end=a('strides')[i])) for i in range(n)]),
            Attr('dilation', [1] * n),
            Attr('data_layout', Var(STR, choices=data_layout_choices)),
            Attr('groups', Var(INT, ran=iran(1, in_chan))),
            Attr('kernel_layout', Var(STR, choices=kernel_layout_choices)),
        ],
        in_num=2,
        in_ranks=[n + 2] * 2,
        in_dtypes=List(2, lambda _: Var()),
        in_shapes=[
            LayoutMap(a('data_layout'), data_chan_first, [Var() for _ in range(n + 2)]),
            LayoutMap(
                a('kernel_layout'), kernel_chan_first,
                Concat([a('channels') // a('groups'), in_chan], a('kernel_size'))
            ),
        ],
        extra=
        [
            a('channels') % a('groups') == 0,
            in_chan % a('groups') == 0,
        ]
        + dims_extra,
        out_num=1,
        out_ranks=[n + 2],
        out_dtypes=[IN[0].dtype],
        out_shapes=[
            LayoutMap(a('data_layout'), data_chan_first, [IN[0].shape[0], a('channels')] + out_dims)
        ]
    )


def _create_conv_trans_nd_no_group(n: int):
    if TypeSpec.for_graph:
        out_dims = [
            _conv_trans_dim(
                IN[0].shape[2 + i], a('kernel_size')[i], a('strides')[i], a('dilation')[i],
                a('padding')[i], a('padding')[n + i], a('output_padding')[i]
            ) for i in range(n)
        ]
        return TypeSpec(
            attrs=[
                Attr('kernel_size', [Var(INT, ran=kernel_ran) for _ in range(n)]),
                Attr('channels', Var(INT, ran=dim_ran)),
                Attr('strides', [Var(INT, ran=stride_ran) for _ in range(n)]),
                Attr('padding', [Var(INT, ran=pad_ran) for _ in range(2 * n)]),
                Attr('output_padding',
                     [Var(INT, ran=Range(end=a('strides')[i])) for i in range(n)]),
                Attr('dilation', [1] * n),
            ],
            in_num=2,
            in_ranks=[n + 2] * 2,
            in_dtypes=List(2, lambda _: Var()),
            in_shapes=[
                [Var() for _ in range(n + 2)],
                Concat([IN[0].shape[1], a('channels')], a('kernel_size')),
            ],
            extra=[d == IN[0].shape[2 + i] for i, d in enumerate(out_dims)],
            out_num=1,
            out_ranks=[n + 2],
            out_dtypes=[IN[0].dtype],
            out_shapes=[
                [IN[0].shape[0], a('channels')] + out_dims
            ]
        )

    # Layout
    dim_str = 'DHW'[-n:]
    kernel_chan_first = 'OI' + dim_str
    kernel_chan_last = dim_str + 'IO'
    kernel_layout_choices = [kernel_chan_first, kernel_chan_last]

    # Dimension
    out_dims = [
        _conv_trans_dim(
            IN[0].shape[2 + i], a('kernel_size')[i],
            a('strides')[i], a('dilation')[i], a('padding')[i], a('padding')[n + i],
            a('output_padding')[i]
        ) for i in range(n)
    ]
    dims_extra = [d >= 1 for d in out_dims]

    return TypeSpec(
        attrs=[
            Attr('kernel_size', [Var(INT, ran=kernel_ran) for _ in range(n)]),
            Attr('channels', Var(INT, ran=dim_ran)),
            Attr('strides', [Var(INT, ran=stride_ran) for _ in range(n)]),
            Attr('padding', [Var(INT, ran=pad_ran) for _ in range(2 * n)]),
            Attr('output_padding', [Var(INT, ran=Range(end=a('strides')[i])) for i in range(n)]),
            Attr('dilation', [1] * n),
            Attr('kernel_layout', Var(STR, choices=kernel_layout_choices)),
        ],
        in_num=2,
        in_ranks=[n + 2] * 2,
        in_dtypes=List(2, lambda _: Var()),
        in_shapes=[
            [Var() for _ in range(n + 2)],
            LayoutMap(
                a('kernel_layout'), kernel_chan_first,
                Concat([IN[0].shape[1], a('channels')], a('kernel_size'))
            ),
        ],
        extra=dims_extra,
        out_num=1,
        out_ranks=[n + 2],
        out_dtypes=[IN[0].dtype],
        out_shapes=[
            [IN[0].shape[0], a('channels')] + out_dims
        ]
    )


Op('nn.conv1d_transpose', lambda: _create_conv_trans_nd_no_group(1), params=[1])
Op('nn.conv2d_transpose', lambda: _create_conv_trans_nd(2), params=[1])
Op('nn.conv3d_transpose', lambda: _create_conv_trans_nd_no_group(3), params=[1])


def _pool_dim(i: Expr, w: Expr, dil: Expr, pad_b: Expr, pad_e: Expr, stride: Expr, ceil: Expr):
    dim_ns = _conv_dim_no_stride(i, w, dil, pad_b, pad_e)
    return Cond(ceil, dim_ns + stride - 1, dim_ns) / stride + 1


def _create_pool_nd(n: int):
    if TypeSpec.for_graph:
        out_dims = [
            _pool_dim(
                IN[0].shape[2 + i], a('pool_size')[i], a('dilation')[i],
                a('padding')[i], a('padding')[n + i], a('strides')[i], a('ceil_mode'),
            ) for i in range(n)
        ]
        return TypeSpec(
            attrs=[
                Attr('pool_size', [Var(INT, ran=kernel_ran) for _ in range(n)]),
                Attr('strides', [Var(INT, ran=stride_ran) for _ in range(n)]),
                Attr('padding', [Var(INT, ran=pad_ran) for _ in range(2 * n)]),
                Attr('dilation', [Var(INT, ran=dil_ran) for _ in range(n)]),
                Attr('ceil_mode', Var(BOOL)),
            ],
            in_num=1,
            in_ranks=[n + 2],
            in_dtypes=[Var()],
            in_shapes=[
                [Var() for _ in range(n + 2)],
            ],
            extra=
            [
                _conv_dim_no_stride(
                    IN[0].shape[2 + i], a('pool_size')[i], a('dilation')[i],
                    a('padding')[i], a('padding')[n + i]
                ) >= 0 for i in range(n)
            ] + [
                (IN[0].shape[2 + i] // 2).max(1) == d for i, d in enumerate(out_dims)
            ],
            out_num=1,
            out_ranks=[n + 2],
            out_dtypes=[IN[0].dtype],
            out_shapes=[
                Concat(IN[0].shape[:2], out_dims)
            ]
        )

    # Layout
    dim_str = 'DHW'[-n:]
    chan_first = 'NC' + dim_str
    chan_last = 'N' + dim_str + 'C'
    layout_choices = [chan_first, chan_last]

    # Dimension
    in_chan = IN[0].shape[LayoutIndex(a('layout'), 'C')]
    dims_extra = [
        _conv_dim_no_stride(
            IN[0].shape[LayoutIndex(a('layout'), dim_str[i])], a('pool_size')[i],
            a('dilation')[i], a('padding')[i], a('padding')[n + i]
        ) >= 0 for i in range(n)
    ]
    out_dims = [
        _pool_dim(
            IN[0].shape[LayoutIndex(a('layout'), dim_str[i])], a('pool_size')[i],
            a('dilation')[i], a('padding')[i], a('padding')[n + i], a('strides')[i],
            a('ceil_mode'),
        ) for i in range(n)
    ]

    return TypeSpec(
        attrs=[
            Attr('pool_size', [Var(INT, ran=kernel_ran) for _ in range(n)]),
            Attr('strides', [Var(INT, ran=stride_ran) for _ in range(n)]),
            Attr('padding', [Var(INT, ran=pad_ran) for _ in range(2 * n)]),
            Attr('dilation', [Var(INT, ran=dil_ran) for _ in range(n)]),
            Attr('layout', Var(STR, choices=layout_choices)),
            Attr('ceil_mode', Var(BOOL)),
        ],
        in_num=1,
        in_ranks=[n + 2],
        in_dtypes=[Var()],
        in_shapes=[
            LayoutMap(a('layout'), chan_first, [Var() for _ in range(n + 2)]),
        ],
        extra=dims_extra,
        out_num=1,
        out_ranks=[n + 2],
        out_dtypes=[IN[0].dtype],
        out_shapes=[
            LayoutMap(a('layout'), chan_first, [IN[0].shape[0], in_chan] + out_dims)
        ]
    )


Op('nn.max_pool1d', lambda: _create_pool_nd(1))
Op('nn.max_pool2d', lambda: _create_pool_nd(2))
Op('nn.max_pool3d', lambda: _create_pool_nd(3))


def _create_avg_pool_nd(n: int):
    spec = _create_pool_nd(n)
    spec.add_attr(Attr('count_include_pad', Var(BOOL)))
    return spec


Op('nn.avg_pool1d', lambda: _create_avg_pool_nd(1))
Op('nn.avg_pool2d', lambda: _create_avg_pool_nd(2))
Op('nn.avg_pool3d', lambda: _create_avg_pool_nd(3))


def _create_global_pool_nd(n: int):
    if TypeSpec.for_graph:
        return TypeSpec(
            attrs=[],
            in_num=1,
            in_ranks=[n + 2],
            in_dtypes=[Var()],
            in_shapes=[
                [Var() for _ in range(n + 2)],
            ],
            extra=[],
            out_num=1,
            out_ranks=[n + 2],
            out_dtypes=[IN[0].dtype],
            out_shapes=[
                Concat(IN[0].shape[:2], [1] * n)
            ]
        )

    # Layout
    dim_str = 'DHW'[-n:]
    chan_first = 'NC' + dim_str
    chan_last = 'N' + dim_str + 'C'
    layout_choices = [chan_first, chan_last]

    # Dimension
    in_chan = IN[0].shape[LayoutIndex(a('layout'), 'C')]

    return TypeSpec(
        attrs=[
            Attr('layout', Var(STR, choices=layout_choices)),
        ],
        in_num=1,
        in_ranks=[n + 2],
        in_dtypes=[Var()],
        in_shapes=[
            LayoutMap(a('layout'), chan_first, [Var() for _ in range(n + 2)]),
        ],
        extra=[],
        out_num=1,
        out_ranks=[n + 2],
        out_dtypes=[IN[0].dtype],
        out_shapes=[
            LayoutMap(a('layout'), chan_first, [IN[0].shape[0], in_chan] + [1] * n)
        ]
    )


Op('nn.global_avg_pool2d', lambda: _create_global_pool_nd(2))


def _create_adapt_pool_nd(n: int):
    if TypeSpec.for_graph:
        return TypeSpec(
            attrs=[
                Attr('output_size', [1] * n),
            ],
            in_num=1,
            in_ranks=[n + 2],
            in_dtypes=[Var()],
            in_shapes=[
                [Var() for _ in range(n + 2)],
            ],
            extra=[],
            out_num=1,
            out_ranks=[n + 2],
            out_dtypes=[IN[0].dtype],
            out_shapes=[
                Concat(IN[0].shape[:2], a('output_size'))
            ]
        )

    # Layout
    dim_str = 'DHW'[-n:]
    chan_first = 'NC' + dim_str
    chan_last = 'N' + dim_str + 'C'
    layout_choices = [chan_first, chan_last]

    # Dimension
    in_chan = IN[0].shape[LayoutIndex(a('layout'), 'C')]

    return TypeSpec(
        attrs=[
            Attr('layout', Var(STR, choices=layout_choices)),
            Attr('output_size',
                 [Var(INT, ran=iran(1, IN[0].shape[LayoutIndex(a('layout'), dim_str[i])]))
                  for i in range(n)]),
        ],
        in_num=1,
        in_ranks=[n + 2],
        in_dtypes=[Var()],
        in_shapes=[
            LayoutMap(a('layout'), chan_first, [Var() for _ in range(n + 2)]),
        ],
        extra=[],
        out_num=1,
        out_ranks=[n + 2],
        out_dtypes=[IN[0].dtype],
        out_shapes=[
            LayoutMap(a('layout'), chan_first,
                      Concat([IN[0].shape[0], in_chan], a('output_size')))
        ]
    )


Op('nn.adaptive_max_pool1d', lambda: _create_adapt_pool_nd(1))
Op('nn.adaptive_max_pool2d', lambda: _create_adapt_pool_nd(2))
Op('nn.adaptive_max_pool3d', lambda: _create_adapt_pool_nd(3))
Op('nn.adaptive_avg_pool1d', lambda: _create_adapt_pool_nd(1))
Op('nn.adaptive_avg_pool2d', lambda: _create_adapt_pool_nd(2))
Op('nn.adaptive_avg_pool3d', lambda: _create_adapt_pool_nd(3))


def _create_upsampling_2d():
    return TypeSpec(
        attrs=[
            Attr('scale_h', Var(INT, choices=[2] if TypeSpec.for_graph else [1, 2])),
            Attr('scale_w', Var(INT, choices=[2] if TypeSpec.for_graph else [1, 2])),
            Attr('layout', 'NCHW' if TypeSpec.for_graph else Var(STR, choices=['NCHW', 'NHWC'])),
            Attr('method', Var(STR, choices=['nearest_neighbor', 'bilinear', 'bicubic'])),
            Attr('align_corners', Var(BOOL)),
        ],
        in_num=1,
        in_ranks=[4],
        in_dtypes=[Var()],
        in_shapes=[
            [Var() for _ in range(4)],
        ],
        extra=[],
        out_num=1,
        out_ranks=[4],
        out_dtypes=[IN[0].dtype],
        out_shapes=[
            LayoutMap(a('layout'), 'NCHW', [
                IN[0].shape[0], IN[0].shape[LayoutIndex(a('layout'), 'C')],
                IN[0].shape[LayoutIndex(a('layout'), 'H')] * a('scale_h'),
                IN[0].shape[LayoutIndex(a('layout'), 'W')] * a('scale_w'),
            ]),
        ]
    )


Op('nn.upsampling', _create_upsampling_2d)


def _create_upsampling_3d():
    return TypeSpec(
        attrs=[
            Attr('scale_d', Var(INT, choices=[2] if TypeSpec.for_graph else [1, 2])),
            Attr('scale_h', Var(INT, choices=[2] if TypeSpec.for_graph else [1, 2])),
            Attr('scale_w', Var(INT, choices=[2] if TypeSpec.for_graph else [1, 2])),
            Attr('layout', 'NCDHW' if TypeSpec.for_graph else Var(STR, choices=['NCDHW', 'NDHWC'])),
            Attr('method', Var(STR, choices=['nearest_neighbor', 'trilinear'])),
            Attr('coordinate_transformation_mode',
                 Var(STR, choices=['half_pixel', 'align_corners', 'asymmetric'])),
        ],
        in_num=1,
        in_ranks=[5],
        in_dtypes=[Var()],
        in_shapes=[
            [Var() for _ in range(5)],
        ],
        extra=[],
        out_num=1,
        out_ranks=[5],
        out_dtypes=[IN[0].dtype],
        out_shapes=[
            LayoutMap(a('layout'), 'NCDHW', [
                IN[0].shape[0], IN[0].shape[LayoutIndex(a('layout'), 'C')],
                IN[0].shape[LayoutIndex(a('layout'), 'D')] * a('scale_d'),
                IN[0].shape[LayoutIndex(a('layout'), 'H')] * a('scale_h'),
                IN[0].shape[LayoutIndex(a('layout'), 'W')] * a('scale_w'),
            ]),
        ]
    )


Op('nn.upsampling3d', _create_upsampling_3d)


def _create_pad():
    pad_width = a('pad_width')
    return TypeSpec(
        attrs=[
            Attr('pad_width',
                 Concat([[0, 0], [0, 0]],
                        List(IN[0].rank - 2, lambda _: [Var(INT, ran=pad_ran, tmpl=True)] * 2))
                 if TypeSpec.for_graph else List(
                     IN[0].rank, lambda _: [Var(INT, ran=pad_ran, tmpl=True)] * 2)),
            Attr('pad_mode', Var(STR, choices=['constant', 'edge'])),
        ],
        in_num=2,
        in_ranks=[Var(ran=Range(3, max_rank)), 0],
        in_dtypes=List(2, lambda _: Var()),
        in_shapes=[List(IN[0].rank, lambda _: Var(tmpl=True)), []],
        extra=[],
        out_num=1,
        out_ranks=[IN[0].rank],
        out_dtypes=[IN[0].dtype],
        out_shapes=[
            List(IN[0].rank, lambda i: IN[0].shape[i] + pad_width[i][0] + pad_width[i][1])
        ]
    )


Op('nn.pad', _create_pad, params=[1])


def _create_norm():
    return TypeSpec(
        attrs=[
            Attr('axis', 1 if TypeSpec.for_graph else Var(INT, ran=Range(end=IN[0].rank))),
            Attr('epsilon', 1e-5),
            Attr('center', Var(BOOL)),
            Attr('scale', Var(BOOL)),
        ],
        in_num=3,
        in_ranks=[Var(ran=dl_rank_ran if TypeSpec.for_graph else rank_ran), 1, 1],
        in_dtypes=List(3, lambda _: Var(choices=float_dtypes)),
        in_shapes=Concat([List(IN[0].rank, lambda _: Var(tmpl=True))],
                         [[IN[0].shape[a('axis')]] for _ in range(2)]),
        extra=[],
        out_num=1,
        out_ranks=[IN[0].rank],
        out_dtypes=[IN[0].dtype],
        out_shapes=[IN[0].shape]
    )


Op('nn.layer_norm', _create_norm, params=[1, 2])


def _create_instance_norm():
    spec = _create_norm()
    if TypeSpec.for_graph:
        spec.in_dtypes = [DataType.f(32)] * 3
        spec.in_ranks = [Var(ran=iran(3, max_rank)), 1, 1]
    return spec


Op('nn.instance_norm', _create_instance_norm, params=[1, 2])


def _create_group_norm():
    spec = _create_norm()
    spec.add_attr(
        Attr('num_groups', Var(INT, ran=iran(1, IN[0].shape[a('axis')])))
    )
    spec.add_extra(IN[0].shape[a('axis')] % a('num_groups') == 0)
    return spec


Op('nn.group_norm', _create_group_norm, params=[1, 2])


def _create_batch_norm():
    return TypeSpec(
        attrs=[
            Attr('axis', 1 if TypeSpec.for_graph else Var(INT, ran=Range(end=IN[0].rank))),
            Attr('epsilon', 1e-5),
            Attr('center', Var(BOOL)),
            Attr('scale', Var(BOOL)),
        ],
        in_num=5,
        in_ranks=[Var(ran=dl_rank_ran if TypeSpec.for_graph else rank_ran), 1, 1, 1, 1],
        in_dtypes=List(5, lambda _: Var(choices=float_dtypes)),
        in_shapes=Concat([List(IN[0].rank, lambda _: Var(tmpl=True))],
                         [[IN[0].shape[a('axis')]] for _ in range(4)]),
        extra=[],
        out_num=3,
        out_ranks=[IN[0].rank, 1, 1],
        out_dtypes=[IN[0].dtype] * 3,
        out_shapes=[IN[0].shape] + [[IN[0].shape[a('axis')]]] * 2
    )


Op('nn.batch_norm', _create_batch_norm, params=[1, 2, 3, 4], ignored_outs=[1, 2])


def _create_dense():
    return TypeSpec(
        attrs=[
            Attr('units', Var(INT, ran=dim_ran)),
        ],
        in_num=2,
        in_ranks=[2, 2],
        in_dtypes=List(2, lambda _: Var()),
        in_shapes=[
            List(IN[0].rank, lambda _: Var(tmpl=True)),
            [a('units'), IN[0].shape[-1]]
        ],
        extra=[],
        out_num=1,
        out_ranks=[IN[0].rank],
        out_dtypes=[IN[0].dtype],
        out_shapes=[
            Concat(IN[0].shape[Range(end=-1)], [a('units')])
        ]
    )


Op('nn.dense', _create_dense, params=[1])


def _create_batch_flatten():
    return TypeSpec(
        attrs=[],
        in_num=1,
        in_ranks=[Var(ran=dl_rank_ran if TypeSpec.for_graph else rank_ran)],
        in_dtypes=[Var()],
        in_shapes=[
            List(IN[0].rank, lambda _: Var(tmpl=True)),
        ],
        extra=[],
        out_num=1,
        out_ranks=[2],
        out_dtypes=[IN[0].dtype],
        out_shapes=[
            [IN[0].shape[0], ReduceArray(IN[0].shape[1:], ArithOp.MUL, 1)]
        ]
    )


Op('nn.batch_flatten', _create_batch_flatten)
