import typing as t
from typing import Dict, TypeVar, Callable, Optional, Iterable, cast, List
from warnings import warn

from .config import params
from .expr import Expr, ExprLike, Var, Range, DataType, BOOL, INT, DTYPE
from .expr.array import Tuple, List
from .expr.basic import ExprKind, Const, to_expr, iran
from .expr.fmt import print_expr
from .expr.infer import ExprTypeError, infer_type
from .expr.ty import Type, ListType, TyVar, common_dtypes
from .util import CodeBuffer, cls_name, unwrap_or

max_in_num = params['spec.max_in_num']
in_num_ran = iran(1, max_in_num)
max_out_num = params['spec.max_out_num']
out_num_ran = iran(1, max_out_num)
max_rank = params['spec.max_rank']
rank_ran = iran(1, max_rank)
dl_rank_ran = iran(2, max_rank)
max_dim = params['spec.max_dim']
dim_ran = iran(1, max_dim)


class Attr:
    """
    Specification of an operator attribute.
    """

    def __init__(self, name: str, expr: ExprLike):
        self.name_ = name
        self.expr_ = to_expr(expr)


class TypeSpec:
    """
    Specification of type constraints for an operator.
    """

    # Whether specification is used for computation graph generation. When specifying for graphs,
    # the space of input types and attributes specified by the constraints is usually more
    # restricted.
    for_graph = True

    def __init__(self,
                 attrs: t.List[Attr],
                 in_num: ExprLike,
                 in_ranks: ExprLike,
                 in_dtypes: ExprLike,
                 in_shapes: ExprLike,
                 extra: t.List[ExprLike],
                 out_num: ExprLike,
                 out_ranks: ExprLike,
                 out_dtypes: ExprLike,
                 out_shapes: ExprLike):
        self.attrs = attrs
        self.in_num = in_num
        self.in_ranks = in_ranks
        self.in_dtypes = in_dtypes
        self.in_shapes = in_shapes
        self.extra = extra
        self.out_num = out_num
        self.out_ranks = out_ranks
        self.out_dtypes = out_dtypes
        self.out_shapes = out_shapes

    @property
    def attrs(self):
        """
        List of attributes of the operator. Necessary type annotation for variables is required
        for type inference. Later attributes can refer to previous ones.
        """
        return self._attrs

    @attrs.setter
    def attrs(self, *v: t.List[Attr]):
        self._attrs = v[0]

    def has_attr(self, name: str):
        return any(a.name_ == name for a in self._attrs)

    def reset_attr(self, new: Attr):
        for a in self._attrs:
            if a.name_ == new.name_:
                a.expr_ = new.expr_
                return
        raise ValueError(f'Attribute \'{new.name_}\' not found.')

    def add_attr(self, new: Attr):
        if self.has_attr(new.name_):
            raise ValueError(f'Attribute \'{new.name_}\' already defined.')
        self._attrs.append(new)

    @property
    def in_num(self):
        """
        Expression which can be evaluated to number of inputs of the operator. Can refer to
        attributes.
        """
        return self._in_num

    @in_num.setter
    def in_num(self, *v: ExprLike):
        self._in_num = to_expr(v[0])

    def in_num_choices(self) -> t.List[int]:
        """
        Possible choices of input number.
        """
        return int_expr_choices(self._in_num, 1, max_in_num + 1)

    @property
    def is_variadic(self):
        return self.in_num.kind != ExprKind.CONST

    @property
    def has_no_input(self):
        return self.in_num.kind == ExprKind.CONST and cast(Const, self.in_num).val_ == 0

    @property
    def in_ranks(self):
        """
        Expression which can be evaluated to the ranks of input tensors. It must be consistent
        with the input number. Can refer to attributes and input number.
        """
        return self._in_ranks

    @in_ranks.setter
    def in_ranks(self, *v: ExprLike):
        self._in_ranks = to_expr(v[0])

    @property
    def first_rank_choices(self) -> t.List[int]:
        """
        Possible choices of first input tensor's rank.
        """
        min_rank = 1
        if self.has_no_input:
            return []
        if self._in_ranks.kind == ExprKind.TUPLE:
            rank_tup = cast(Tuple, self._in_ranks)
            first_rank = rank_tup.fields_[0]
            return int_expr_choices(first_rank, min_rank, max_rank + 1)
        elif self._in_ranks.kind == ExprKind.LIST:
            rank_lst = cast(List, self._in_ranks)
            return int_expr_choices(rank_lst.body_, min_rank, max_rank + 1)
        else:
            return list(range(min_rank, max_rank + 1))

    @property
    def in_dtypes(self):
        """
        Expression which can be evaluated to the data types of input tensors. It must be
        consistent with the input number. Can refer to attributes and input number.
        """
        return self._in_dtypes

    @in_dtypes.setter
    def in_dtypes(self, *v: ExprLike):
        self._in_dtypes = to_expr(v[0])

    @property
    def first_dtype_choices(self) -> t.List[DataType]:
        if self.has_no_input:
            return []
        if self._in_dtypes.kind == ExprKind.TUPLE:
            tup = cast(Tuple, self._in_dtypes)
            return expr_choices(tup.fields_[0], common_dtypes)
        elif self._in_dtypes.kind == ExprKind.LIST:
            lst = cast(List, self._in_dtypes)
            return expr_choices(lst.body_, common_dtypes)
        else:
            return common_dtypes

    @property
    def in_shapes(self):
        """
        Expression which can be evaluated to the shapes of input tensors. It must be consistent
        with the input number and input ranks. Can refer to attributes, input number and input
        ranks.
        """
        return self._in_shapes

    @in_shapes.setter
    def in_shapes(self, *v: ExprLike):
        self._in_shapes = to_expr(v[0])

    @property
    def extra(self):
        """
        List of extra constraints on attributes and inputs.
        """
        return self._extra

    @extra.setter
    def extra(self, *v: t.List[ExprLike]):
        self._extra = [to_expr(c) for c in v[0]]

    def add_extra(self, e: Expr):
        self._extra.append(e)

    @property
    def out_num(self):
        """
        Expression which can be evaluated to number of outputs of the operator. Can refer to
        attributes and inputs.
        """
        return self._out_num

    @out_num.setter
    def out_num(self, *v: ExprLike):
        self._out_num = to_expr(v[0])

    @property
    def out_ranks(self):
        """
        Expression which can be evaluated to the ranks of output tensors. It must be consistent
        with the output number. Can refer to attributes, inputs and output number.
        """
        return self._out_ranks

    @out_ranks.setter
    def out_ranks(self, *v: ExprLike):
        self._out_ranks = to_expr(v[0])

    @property
    def out_dtypes(self):
        """
        Expression which can be evaluated to the data types of output tensors. It must be
        consistent with the output number. Can refer to attributes, inputs and output number.
        """
        return self._out_dtypes

    @out_dtypes.setter
    def out_dtypes(self, *v: ExprLike):
        self._out_dtypes = to_expr(v[0])

    @property
    def out_shapes(self):
        """
        Expression which can be evaluated to the shapes of output tensors. It must be consistent
        with the output number and output ranks. Can refer to attributes, inputs, output number
        and output ranks.
        """
        return self._out_shapes

    @out_shapes.setter
    def out_shapes(self, *v: ExprLike):
        self._out_shapes = to_expr(v[0])

    def check(self):
        """
        Perform type inference on the specified constraints. The check should be done after all the
        constraints of an operator are determined.
        """
        # Attribute types
        attr_ty: Dict[str, Type] = {}
        for attr in self.attrs:
            ty = self._infer_type(attr.expr_, attr_ty, f'attr[\'{attr.name_}\']')
            attr_ty[attr.name_] = ty

        # Inputs
        self._infer_type(self.in_num, attr_ty, 'in_num', INT)
        self._infer_type(self.in_ranks, attr_ty, 'in_ranks', ListType(INT))
        self._infer_type(self.in_dtypes, attr_ty, 'in_dtypes', ListType(DTYPE))
        self._infer_type(self.in_shapes, attr_ty, 'in_shapes', ListType(ListType(INT)))

        # Extra constraints
        for i, cmp in enumerate(self.extra):
            self._infer_type(cmp, attr_ty, f'extra[{i}]', BOOL)

        # Outputs
        self._infer_type(self.out_num, attr_ty, 'out_num', INT)
        self._infer_type(self.out_ranks, attr_ty, 'out_ranks', ListType(INT))
        self._infer_type(self.out_dtypes, attr_ty, 'out_dtypes', ListType(DTYPE))
        self._infer_type(self.out_shapes, attr_ty, 'out_shapes', ListType(ListType(INT)))

    @staticmethod
    def _infer_type(expr: Expr, attr_ty: Dict[str, Type], name: str, hint: Type = TyVar()):
        try:
            ty = infer_type(expr, attr_ty, hint=hint)
        except ExprTypeError as err:
            buf = CodeBuffer()
            print_expr(expr, buf, [err.expr_])
            raise SpecCheckError(name, err.msg_, str(buf))
        return ty

    def __str__(self):
        buf = CodeBuffer()
        buf.write(cls_name(self))

        def print_fn(e: Expr):
            return lambda: print_expr(e, buf, [])

        buf.write_named_multi([
            ('attr', lambda: buf.write_named_multi(
                [(a.name_, print_fn(a.expr_)) for a in self.attrs],
                prefix='[', suffix=']'
            )),
            ('in_num', print_fn(self.in_num)),
            ('in_ranks', print_fn(self.in_ranks)),
            ('in_dtypes', print_fn(self.in_dtypes)),
            ('in_shapes', print_fn(self.in_shapes)),
            ('extra', lambda: buf.write_pos_multi(
                [print_fn(cmp) for cmp in self.extra],
                prefix='[', suffix=']'
            )),
            ('out_num', print_fn(self.out_num)),
            ('out_ranks', print_fn(self.out_ranks)),
            ('out_dtypes', print_fn(self.out_dtypes)),
            ('out_shapes', print_fn(self.out_shapes)),
        ])

        return str(buf)


T = TypeVar('T')


def expr_choices(e: Expr, default: Iterable[T]) -> t.List[T]:
    if e.kind == ExprKind.CONST:
        return [cast(Const, e).val_]
    elif e.kind == ExprKind.VAR:
        var = cast(Var, e)
        if var.choices_ is not None:
            return extract_const_choices(cast(Var, e).choices_, default)
        elif var.type_ == BOOL:
            return [True, False]
        elif var.type_ == INT and var.ran_ is not None:
            return int_range_choices(var.ran_, 0, max_dim + 1)
        else:
            return list(default)
    else:
        return list(default)


def extract_const_choices(e: Expr, default: Iterable[T]) -> t.List[T]:
    if e.kind != ExprKind.TUPLE:
        return list(default)
    tup = cast(Tuple, e)
    return [cast(Const, f).val_ for f in tup.fields_ if f.kind == ExprKind.CONST]


def int_expr_choices(e: Expr, begin: int, end: int) -> t.List[int]:
    if e.kind == ExprKind.CONST:
        return [cast(Const, e).val_]
    elif e.kind == ExprKind.VAR:
        var = cast(Var, e)
        if var.ran_ is not None:
            return int_range_choices(var.ran_, begin, end)
        elif var.choices_ is not None:
            return extract_const_choices(e, range(begin, end))
    return list(range(begin, end))


def int_range_choices(ran: Range, begin: int, end: int) -> t.List[int]:
    if ran.begin_.kind == ExprKind.CONST:
        begin = max(begin, cast(Const, ran.begin_).val_)
    if ran.end_.kind == ExprKind.CONST:
        end = min(end, cast(Const, ran.end_).val_)
    return list(range(begin, end))


class SpecCheckError(Exception):
    def __init__(self, name: str, msg: str, code: str):
        self.name_ = name
        self.msg_ = msg
        self.code_ = code

    def __str__(self):
        return f'Specification error in {self.name_}: {self.msg_}\n' \
               f'{self.code_}'


class Op:
    """
    Operator.
    """

    def __init__(self, name: str, spec_f: Callable[[], TypeSpec],
                 params: Optional[t.List[int]] = None, ignored_outs: Optional[t.List[int]] = None,
                 register: bool = True):
        self.name_ = name
        self.spec_f_ = spec_f
        self.params_ = unwrap_or(params, [])
        self.ignored_ = unwrap_or(ignored_outs, [])
        if register:
            OpRegistry.register(self)

    @property
    def spec(self):
        spec = self.spec_f_()
        try:
            spec.check()
        except SpecCheckError as err:
            raise RuntimeError(
                f'Specification check failed of operator \'{self.name_}\': '
                f'{err.msg_}\n'
                f'{err.name_}={err.code_}'
            )
        return spec

    def __repr__(self):
        return self.name_


class OpRegistry:
    """
    Registry for all defined operators.
    """

    _ops: t.List[Op] = []
    _table: Dict[str, Op] = {}

    @classmethod
    def register(cls, op: Op):
        if op.name_ in cls._table:
            warn(f'Operator {op.name_} has already been registered.')
        else:
            cls._ops.append(op)
            cls._table[op.name_] = op

    @classmethod
    def get(cls, name: str):
        if name not in cls._table:
            raise ValueError(f'Operator {name} not found.')
        return cls._table[name]

    @classmethod
    def names(cls):
        return cls._table.keys()

    @classmethod
    def ops(cls):
        return iter(cls._ops)
