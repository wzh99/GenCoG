if __name__ == '__main__':
    from typegraph import *

    m = IN[0].rank
    n = IN[1].rank
    bcast = ConstraintSet(
        attrs=[],
        in_num=2,
        in_ranks=[Var(Int()), Var(Int())],
        in_dtypes=[Var(DType()), Var(DType())],
        in_shapes=[
            List(m, lambda _: Var(Int())),
            List(n, lambda _: Var(Int())),
        ],
        extra=[
            IN[0].dtype == IN[1].dtype,
            ForEach(Range(end=m.min(n)), lambda i: Or(
                IN[0].shape[m - i] == IN[1].shape[n - i],
                IN[0].shape[m - i] == 1,
                IN[1].shape[n - i] == 1,
            )),
        ],
        out_num=1,
        out_ranks=[IN[0].rank.max(IN[1].rank)],
        out_dtypes=[IN[0].dtype],
        out_shapes=[
            Cond(
                m >= n,
                Concat(
                    IN[0].shape[Range(end=m - n)],
                    List(n, lambda i: IN[0].shape[m - n + i].max(IN[1].shape[i])),
                ),
                Concat(
                    IN[1].shape[Range(end=n - m)],
                    List(m, lambda i: IN[0].shape[i].max(IN[1].shape[n - m + 1])),
                )
            )
        ],
    )
    pass
