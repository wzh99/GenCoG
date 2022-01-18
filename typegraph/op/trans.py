from ..expr import *
from ..spec import Attr, ConstraintSet, Op

_concat = ConstraintSet(
    attrs=[
        Attr('axis', Var(t=INT, ran=Range(0, IN[0].rank)))
    ],
    in_num=Var(),
    in_ranks=Concat([Var(ran=Range(begin=1))], List(IN.num - 1, lambda _: IN[0].rank)),
    in_dtypes=Concat([Var()], List(IN.num - 1, lambda _: IN[0].dtype)),
    in_shapes=Concat(
        [List(IN[0].rank, lambda _: Var())],
        List(
            IN.num - 1,
            lambda _: List(
                IN[0].rank,
                lambda j: Cond(j == a('axis'), Var(), IN[0].shape[j]))
        )
    ),
    extra=[],
    out_num=1,
    out_ranks=[IN[0].rank],
    out_dtypes=[IN[0].dtype],
    out_shapes=[
        List(
            IN[0].rank,
            lambda j: Cond(
                j == a('axis'),
                ReduceIndex(IN.num, ArithOp.ADD, lambda i: IN[i].shape[j], 0),
                IN[0].shape[j]
            )
        )
    ]
)

Op('concatenate', _concat)

_split = ConstraintSet(
    attrs=[
        Attr('axis', Var(t=INT, ran=Range(0, IN[0].rank))),
        Attr('indices_or_sections',
             List(Var(ran=Range(begin=1, end=IN[0].shape[a('axis')])), lambda _: Var(INT)))
    ],
    in_num=1,
    in_ranks=[Var(ran=Range(begin=1))],
    in_dtypes=[Var()],
    in_shapes=[List(IN[0].rank, lambda _: Var())],
    extra=[],
    out_num=Len(a('indices_or_sections')) + 1,
    out_ranks=List(OUT.num, lambda _: IN[0].rank),
    out_dtypes=List(OUT.num, lambda _: IN[0].dtype),
    out_shapes=List(
        OUT.num,
        lambda i: List(
            IN[0].rank,
            lambda j: Cond(
                j == a('axis'),
                Cond(
                    i == 0,
                    a('indices_or_sections')[0],
                    Cond(
                        i == Len(a('indices_or_sections')),
                        IN[0].shape[j] - a('indices_or_sections')[-1],
                        a('indices_or_sections')[i] - a('indices_or_sections')[i - 1]
                    )
                ),
                IN[0].shape[j]
            ))
    )
)

Op('split', _split)
