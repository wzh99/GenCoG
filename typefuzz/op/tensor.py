from ..config import params
from ..expr import *
from ..spec import Attr, TypeSpec, Op, in_num_ran, rank_ran, max_rank, dim_ran, dl_rank_ran, \
    max_out_num


def _create_reduce():
    indices = List(IN[0].rank, lambda i: i)

    def is_reduce_axis(i: ExprLike):
        return InSet(i, a('axis')) ^ a('exclude')

    return TypeSpec(
        attrs=[
            Attr('axis', List(Var(), lambda _: Var(INT, tmpl=True))),
            Attr('keepdims', Var(BOOL)),
            Attr('exclude', False if TypeSpec.for_graph else Var(BOOL))
        ],
        in_num=1,
        in_ranks=[Var()],
        in_dtypes=[Var()],
        in_shapes=[
            List(IN[0].rank, lambda _: Var(tmpl=True))
        ],
        extra=[
            Subset(a('axis'), indices[1:] if TypeSpec.for_graph else indices),
            Len(a('axis')) > 0 if TypeSpec.for_graph else True,
        ],
        out_num=1,
        out_ranks=[
            Cond(a('keepdims'), IN[0].rank,
                 Cond(a('exclude'), Len(a('axis')), IN[0].rank - Len(a('axis'))))
        ],
        out_dtypes=[IN[0].dtype],
        out_shapes=[
            Cond(
                a('keepdims'),
                List(IN[0].rank, lambda i: Cond(is_reduce_axis(i), 1, IN[0].shape[i])),
                Map(Filter(indices, lambda i: Not(is_reduce_axis(i))), lambda i: IN[0].shape[i])
            )
        ]
    )


Op('sum', _create_reduce)
Op('mean', _create_reduce)
Op('min', _create_reduce)
Op('max', _create_reduce)


def _create_expand_dims():
    return TypeSpec(
        attrs=[
            Attr('axis', Var(INT, ran=Range(1 if TypeSpec.for_graph else 0, IN[0].rank))),
            Attr('num_newaxis', Var(INT, ran=iran(0, max_rank - IN[0].rank))),
        ],
        in_num=1,
        in_ranks=[Var()],
        in_dtypes=[Var()],
        in_shapes=[List(IN[0].rank, lambda _: Var(tmpl=True))],
        extra=[],
        out_num=1,
        out_ranks=[IN[0].rank + a('num_newaxis')],
        out_dtypes=[IN[0].dtype],
        out_shapes=[
            Concat(
                IN[0].shape[Range(end=a('axis'))],
                List(a('num_newaxis'), lambda _: 1),
                IN[0].shape[Range(a('axis'), IN[0].rank)]
            )
        ]
    )


Op('expand_dims', _create_expand_dims)


def _create_squeeze():
    indices = List(IN[0].rank, lambda i: i)
    return TypeSpec(
        attrs=[
            Attr('axis', List(Var(), lambda _: Var(INT, tmpl=True))),
        ],
        in_num=1,
        in_ranks=[Var()],
        in_dtypes=[Var()],
        in_shapes=[List(IN[0].rank, lambda _: Var(tmpl=True))],
        extra=[
            Subset(a('axis'), Filter(indices[2:] if TypeSpec.for_graph else indices,
                                     lambda i: IN[0].shape[i] == 1)),
            Len(a('axis')) > 0,
        ],
        out_num=1,
        out_ranks=[IN[0].rank - Len(a('axis'))],
        out_dtypes=[IN[0].dtype],
        out_shapes=[
            Map(Filter(indices, lambda i: Not(InSet(i, a('axis')))), lambda i: IN[0].shape[i])
        ]
    )


Op('squeeze', _create_squeeze)


def _create_reshape():
    return TypeSpec(
        attrs=[
            Attr('newshape', List(Var(ran=dl_rank_ran if TypeSpec.for_graph else rank_ran),
                                  lambda _: Var(INT, ran=dim_ran, tmpl=True))),
        ],
        in_num=1,
        in_ranks=[Var()],
        in_dtypes=[Var()],
        in_shapes=[List(IN[0].rank, lambda _: Var(tmpl=True))],
        extra=[
            And(
                a('newshape')[0] == IN[0].shape[0],
            ) if TypeSpec.for_graph else True,
            ReduceArray(a('newshape'), ArithOp.MUL, 1) == ReduceArray(IN[0].shape, ArithOp.MUL, 1)
        ],
        out_num=1,
        out_ranks=[Len(a('newshape'))],
        out_dtypes=[IN[0].dtype],
        out_shapes=[a('newshape')]
    )


Op('reshape', _create_reshape)


def _create_transpose():
    indices = List(IN[0].rank, lambda i: i)
    return TypeSpec(
        attrs=[
            Attr('axes', List(IN[0].rank, lambda _: Var(INT, tmpl=True))),
        ],
        in_num=1,
        in_ranks=[Var()],
        in_dtypes=[Var()],
        in_shapes=[List(IN[0].rank, lambda _: Var(tmpl=True))],
        extra=[
            And(
                a('axes')[0] == 0,
                Perm(a('axes')[1:], indices[1:])
            ) if TypeSpec.for_graph else
            Perm(a('axes'), indices)
        ],
        out_num=1,
        out_ranks=[IN[0].rank],
        out_dtypes=[IN[0].dtype],
        out_shapes=[
            Map(a('axes'), lambda i: IN[0].shape[i])
        ]
    )


Op('transpose', _create_transpose)


def _create_concat():
    return TypeSpec(
        attrs=[
            Attr('axis', 1 if TypeSpec.for_graph else Var(ty=INT, ran=Range(0, IN[0].rank)))
        ],
        in_num=Var(ran=in_num_ran),
        in_ranks=List(IN.num,
                      lambda _: Var(ran=dl_rank_ran if TypeSpec.for_graph else rank_ran)),
        in_dtypes=List(IN.num, lambda _: Var()),
        in_shapes=Concat(
            [List(IN[0].rank, lambda _: Var(ran=dim_ran, tmpl=True))],
            List(IN.num - 1, lambda _: List(IN[0].rank, lambda j: Cond(
                j == a('axis'), Var(ran=dim_ran, tmpl=True), IN[0].shape[j]
            )))
        ),
        extra=[],
        out_num=1,
        out_ranks=[IN[0].rank],
        out_dtypes=[IN[0].dtype],
        out_shapes=[
            List(IN[0].rank, lambda j: Cond(
                j == a('axis'),
                ReduceRange(Range(end=IN.num), ArithOp.ADD, 0, lambda i: IN[i].shape[j]),
                IN[0].shape[j]
            ))
        ]
    )


Op('concatenate', _create_concat)


def _create_split():
    ind = a('indices_or_sections')
    return TypeSpec(
        attrs=[
            Attr('axis', 1 if TypeSpec.for_graph else Var(ty=INT, ran=Range(0, IN[0].rank))),
            Attr('indices_or_sections',
                 List(Var(ran=Range(begin=0,
                                    end=IN[0].shape[a('axis')].min(max_out_num))),
                      lambda _: Var(INT, tmpl=True)))
        ],
        in_num=1,
        in_ranks=[Var(ran=dl_rank_ran if TypeSpec.for_graph else rank_ran)],
        in_dtypes=[Var()],
        in_shapes=[List(IN[0].rank, lambda _: Var(tmpl=True))],
        extra=[
            Cond(
                Len(ind) == 0, True,
                And(
                    ind[0] > 0,
                    ForAll(Range(1, Len(ind)), lambda i: ind[i - 1] < ind[i]),
                    ind[-1] < IN[0].shape[a('axis')]
                ),
            ),
        ],
        out_num=Len(ind) + 1,
        out_ranks=List(OUT.num, lambda _: IN[0].rank),
        out_dtypes=List(OUT.num, lambda _: IN[0].dtype),
        out_shapes=Cond(
            Len(ind) == 0,
            [IN[0].shape],
            List(OUT.num, lambda i: List(IN[0].rank, lambda j: Cond(
                j == a('axis'),
                Cond(i == 0, ind[0],
                     Cond(i == Len(ind), IN[0].shape[j] - ind[-1],
                          ind[i] - ind[i - 1])),
                IN[0].shape[j]
            )))
        )
    )


Op('split', _create_split)


def _create_strided_slice():
    stride_ran = iran(1, params['op.max_stride'])
    indices = List(IN[0].rank, lambda i: i)
    num_axes = Len(a('axes'))
    begin = a('begin')
    end = a('end')
    strides = a('strides')
    axes = a('axes')

    return TypeSpec(
        attrs=[
            Attr('axes', indices[2:] if TypeSpec.for_graph else List(
                Var(), lambda _: Var(INT, tmpl=True))),
            Attr('begin', List(num_axes, lambda i: Var(INT, ran=Range(end=IN[0].shape[axes[i]]),
                                                       tmpl=True))),
            Attr('end', List(num_axes, lambda i: Var(INT, ran=iran(1, IN[0].shape[axes[i]]),
                                                     tmpl=True))),
            Attr('strides', List(num_axes, lambda _: Var(INT, ran=stride_ran, tmpl=True))),
        ],
        in_num=1,
        in_ranks=[Var(ran=dl_rank_ran if TypeSpec.for_graph else rank_ran)],
        in_dtypes=[Var()],
        in_shapes=[List(IN[0].rank, lambda _: Var(tmpl=True))],
        extra=[
            ForAll(Range(end=num_axes),
                   lambda i: (end[i] - begin[i] + strides[i] - 1) / strides[i] == (
                           IN[0].shape[i + 2] / 2).max(1)),
            Len(a('axes')) > 0,
        ] if TypeSpec.for_graph else [
            Subset(a('axes'), indices),
            ForAll(Range(end=num_axes), lambda i: end[i] - begin[i] > 0)
        ],
        out_num=1,
        out_ranks=[IN[0].rank],
        out_dtypes=[IN[0].dtype],
        out_shapes=[
            List(IN[0].rank, lambda i: Cond(
                InSet(i, a('axes')),
                Map(Filter(List(num_axes, lambda j: j), lambda j: axes[j] == i),
                    lambda j: (end[j] - begin[j] + strides[j] - 1) / strides[j])[0],
                IN[0].shape[i]
            ))
        ]
    )


Op('strided_slice', _create_strided_slice)
