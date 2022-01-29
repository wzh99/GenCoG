from ..expr import *
from ..spec import Attr, ConstraintSpec, Op


def _create_reduce():
    ind = List(IN[0].rank, lambda i: i)

    def _is_reduce_axis(i: ExprLike):
        return InSet(i, a('axis')) ^ a('exclude')

    return ConstraintSpec(
        attrs=[
            Attr('axis', List(Var(), lambda _: Var(INT, tmpl=True))),
            Attr('keepdims', Var(BOOL)),
            Attr('exclude', Var(BOOL))
        ],
        in_num=1,
        in_ranks=[Var()],
        in_dtypes=[Var()],
        in_shapes=[
            List(IN[0].rank, lambda _: Var())
        ],
        extra=[
            Subset(a('axis'), ind),
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
                List(IN[0].rank, lambda i: Cond(_is_reduce_axis(i), 1, IN[0].shape[i])),
                Map(Filter(ind, lambda i: _is_reduce_axis(i)), lambda i: IN[0].shape[i])
            )
        ]
    )


Op('sum', _create_reduce())
