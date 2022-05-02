from ..expr import *
from ..spec import TypeSpec, Op, rank_ran, dim_ran


def create_identity():
    return TypeSpec(
        attrs=[],
        in_num=1,
        in_ranks=[Var()],
        in_dtypes=[Var()],
        in_shapes=[List(IN[0].rank, lambda _: Var(tmpl=True))],
        extra=[],
        out_num=1,
        out_ranks=[IN[0].rank],
        out_dtypes=[IN[0].dtype],
        out_shapes=[IN[0].shape]
    )


Op('negative', create_identity)
Op('abs', create_identity)
Op('ceil', create_identity)
Op('floor', create_identity)
Op('round', create_identity)
Op('trunc', create_identity)
Op('exp', create_identity)
Op('sin', create_identity)
Op('cos', create_identity)
Op('tan', create_identity)
Op('sigmoid', create_identity)
Op('tanh', create_identity)


def _create_bcast():
    m = IN[0].rank
    n = IN[1].rank
    if TypeSpec.for_graph:
        return TypeSpec(
            attrs=[],
            in_num=2,
            in_ranks=[Var(), Var(ran=iran(2, m))],
            in_dtypes=List(2, lambda _: Var()),
            in_shapes=[
                List(m, lambda _: Var(ran=dim_ran, tmpl=True)),
                List(n, lambda _: Var(ran=dim_ran, tmpl=True))
            ],
            extra=[
                ForAll(Range(end=n), lambda i: Or(
                    IN[0].shape[m - i - 1] == IN[1].shape[n - i - 1],
                    IN[0].shape[m - i - 1] == 1,
                    IN[1].shape[n - i - 1] == 1,
                ))
            ],
            out_num=1,
            out_ranks=[m],
            out_dtypes=[IN[0].dtype],
            out_shapes=[Concat(
                IN[0].shape[Range(end=m - n)],
                List(n, lambda i: IN[0].shape[m - n + i].max(IN[1].shape[i]))
            )],
        )
    return TypeSpec(
        attrs=[],
        in_num=2,
        in_ranks=List(2, lambda _: Var(ran=rank_ran, tmpl=True)),
        in_dtypes=List(2, lambda _: Var()),
        in_shapes=[
            List(m, lambda _: Var(ran=dim_ran, tmpl=True)),
            List(n, lambda _: Var(ran=dim_ran, tmpl=True))
        ],
        extra=[
            ForAll(Range(end=m.min(n)), lambda i: Or(
                IN[0].shape[m - i - 1] == IN[1].shape[n - i - 1],
                IN[0].shape[m - i - 1] == 1,
                IN[1].shape[n - i - 1] == 1
            ))
        ],
        out_num=1,
        out_ranks=[m.max(n)],
        out_dtypes=[IN[0].dtype],
        out_shapes=[
            Cond(
                m >= n,
                Concat(
                    IN[0].shape[Range(end=m - n)],
                    List(n, lambda i: IN[0].shape[m - n + i].max(IN[1].shape[i]))
                ),
                Concat(
                    IN[1].shape[Range(end=n - m)],
                    List(m, lambda i: IN[0].shape[i].max(IN[1].shape[n - m + i]))
                )
            )
        ]
    )


Op('add', _create_bcast)
Op('subtract', _create_bcast)
Op('multiply', _create_bcast)
Op('divide', _create_bcast)
Op('maximum', _create_bcast)
Op('minimum', _create_bcast)
