from ..expr import *
from ..spec import Attr, TypeSpec, Op, rank_ran, dim_ran


def _create_ew():
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


Op('exp', _create_ew)
Op('nn.relu', _create_ew)


def _create_leaky_relu():
    spec = _create_ew()
    spec.add_attr(Attr('alpha', Var(FLOAT, ran=Range(0., 1.))))
    return spec


Op('nn.leaky_relu', _create_leaky_relu)


def _create_prelu():
    spec = _create_ew()
    spec.add_attr(Attr('axis', Var(INT, ran=Range(end=IN[0].rank))))
    spec.in_num = 2
    spec.in_ranks = [Var(), 1]
    spec.in_dtypes = List(2, lambda _: Var())
    spec.in_shapes = [
        List(IN[0].rank, lambda _: Var(tmpl=True)),
        [IN[0].shape[a('axis')]]
    ]
    return spec


Op('nn.prelu', _create_prelu)


def _create_bcast():
    m = IN[0].rank
    n = IN[1].rank
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


def _create_bcast_cmp():
    cmp = _create_bcast()
    cmp.out_dtypes = [DataType.b()]
    return cmp


Op('less', _create_bcast_cmp)
