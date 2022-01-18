from ..expr import *
from ..spec import ConstraintSet, Op


def _create_bcast():
    m = IN[0].rank
    n = IN[1].rank
    return ConstraintSet(
        attrs=[],
        in_num=2,
        in_ranks=[Var(INT), Var(INT)],
        in_dtypes=[Var(DTYPE), Var(DTYPE)],
        in_shapes=[
            List(m, lambda _: Var(INT)),
            List(n, lambda _: Var(INT))
        ],
        extra=[
            IN[0].dtype == IN[1].dtype,
            ForEach(Range(end=m.min(n)), lambda i: Or(
                IN[0].shape[m - i] == IN[1].shape[n - i],
                IN[0].shape[m - i] == 1,
                IN[1].shape[n - i] == 1
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