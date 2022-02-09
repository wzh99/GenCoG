from ..expr import *
from ..spec import ConstraintSpec, Op, rank_ran, dim_ran


def _create_bcast():
    m = IN[0].rank
    n = IN[1].rank
    return ConstraintSpec(
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


_bcast = _create_bcast()

Op('add', _bcast)


def create_bcast_cmp():
    cmp = _create_bcast()
    cmp.out_dtypes = [DataType.b()]
    return cmp


_bcast_cmp = create_bcast_cmp()

Op('less', _bcast_cmp)
