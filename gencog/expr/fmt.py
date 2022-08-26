import typing as t
from typing import Any, Callable

from colorama import Fore, Back

from .array import Tuple, List, GetItem, Len, Concat, Slice, Map, ReduceArray, ReduceRange, \
    Filter, InSet, Subset, Perm
from .basic import Expr, Const, Var, Range, Symbol, Arith, Cmp, Not, And, Or, ForAll, Cond, \
    GetAttr, Dummy
from .tensor import Num, TensorDesc, Shape, Rank, GetDType, LayoutMap, LayoutIndex
from .visitor import ExprVisitor
from ..util import CodeBuffer, Ref, cls_name, colored_text


class ExprPrinter(ExprVisitor[None, Any]):
    def __init__(self, buf: CodeBuffer, highlights: t.List[Expr]):
        super().__init__()
        self._buf = buf
        self._high = set(Ref(e) for e in highlights)
        self._color_idx = 0

    def visit(self, expr: Expr, env: None):
        high = Ref(expr) in self._high
        if high:
            self._buf.write(Back.RED + Fore.BLACK)
        super().visit(expr, env)
        if high:
            self._buf.write(Back.RESET + Fore.RESET)

    def visit_const(self, const: Const, env: None):
        self._buf.write(repr(const.val_))

    _id_mask = (1 << 16) - 1

    def visit_var(self, var: Var, env: None):
        self._write_cls(var)
        items: t.List[t.Tuple[str, Callable[[], None]]] = [
            ('id', lambda: self._buf.write(hex(id(var) & self._id_mask))),
            ('tmpl', lambda: self._buf.write(str(var.tmpl_))),
        ]
        if var.type_ is not None:
            items.append(('ty', lambda: self._buf.write(str(var.type_))))
        if var.ran_ is not None:
            items.append(('ran', lambda: self.visit(var.ran_, env)))
        if var.choices_ is not None:
            items.append(('choices', lambda: self.visit(var.choices_, env)))
        self._write_named(items)

    def visit_range(self, ran: Range, env: None):
        self._write_cls(ran)
        items: t.List[t.Tuple[str, Callable[[], None]]] = []
        if ran.begin_ is not None:
            items.append(('begin', lambda: self.visit(ran.begin_, env)))
        if ran.end_ is not None:
            items.append(('end', lambda: self.visit(ran.end_, env)))
        self._write_named(items)

    def visit_symbol(self, sym: Symbol, env: None):
        self._buf.write(f'{cls_name(sym)}(id={hex(id(sym) & self._id_mask)})')

    colors = [
        Fore.YELLOW,
        Fore.GREEN,
        Fore.BLUE,
    ]

    def _next_color(self):
        color = self.colors[self._color_idx]
        self._color_idx = (self._color_idx + 1) % len(self.colors)
        return color

    def visit_arith(self, arith: Arith, env: None):
        color = self._next_color()
        self._write_pos(
            [
                lambda: self.visit(arith.lhs_, env),
                lambda: self._buf.write(arith.op_.value),
                lambda: self.visit(arith.rhs_, env)
            ],
            sep=' ',
            prefix=colored_text('(', color),
            suffix=colored_text(')', color)
        )

    def visit_cmp(self, cmp: Cmp, env: None):
        color = self._next_color()
        self._write_pos(
            [
                lambda: self.visit(cmp.lhs_, env),
                lambda: self._buf.write(cmp.op_.value),
                lambda: self.visit(cmp.rhs_, env)
            ],
            sep=' ',
            prefix=colored_text('(', color),
            suffix=colored_text(')', color)
        )

    def visit_not(self, n: Not, env: None):
        self._write_cls(n)
        self._write_pos([n.prop_, lambda prop: self.visit(prop, env)])

    def visit_and(self, a: And, env: None):
        self._write_cls(a)
        self._write_pos_multi(list(map(lambda c: lambda: self.visit(c, env), a.clauses_)))

    def visit_or(self, o: Or, env: None):
        self._write_cls(o)
        self._write_pos_multi(list(map(lambda c: lambda: self.visit(c, env), o.clauses_)))

    def visit_forall(self, forall: ForAll, env: None):
        self._write_cls(forall)
        self._write_named_multi([
            ('ran', lambda: self.visit(forall.ran_, env)),
            ('idx', lambda: self.visit_symbol(forall.idx_, env)),
            ('body', lambda: self.visit(forall.body_, env))
        ])

    def visit_cond(self, cond: Cond, env: None):
        self._write_cls(cond)
        self._write_named_multi([
            ('pred', lambda: self.visit(cond.pred_, env)),
            ('tr_br', lambda: self.visit(cond.tr_br_, env)),
            ('fls_br', lambda: self.visit(cond.fls_br_, env))
        ])

    def visit_attr(self, attr: GetAttr, env: None):
        self._buf.write(f'a(\'{attr.name_}\')')

    def visit_dummy(self, dum: Dummy, env: None):
        self._buf.write(f'{cls_name(dum)}()')

    def visit_num(self, num: Num, env: None):
        self._buf.write(f'{num.t_kind_.value}.num')

    def visit_shape(self, shape: Shape, env: None):
        self._write_tensor_attr(shape.tensor_, 'shape', env)

    def visit_rank(self, rank: Rank, env: None):
        self._write_tensor_attr(rank.tensor_, 'rank', env)

    def visit_dtype(self, dtype: GetDType, env: None):
        self._write_tensor_attr(dtype.tensor_, 'dtype', env)

    def _write_tensor_attr(self, desc: TensorDesc, name: str, env: None):
        self._buf.write(f'{desc.kind_.value}[')
        self.visit(desc.idx_, env)
        self._buf.write(f'].{name}')

    def visit_layout_index(self, i: LayoutIndex, env: None):
        self._write_cls(i)
        self._write_named_multi([
            ('layout', lambda: self.visit(i.layout_, env)),
            ('dim', lambda: self.visit(i.dim_, env)),
        ])

    def visit_layout_map(self, m: LayoutMap, env: None):
        self._write_cls(m)
        self._write_named_multi([
            ('tgt', lambda: self.visit(m.tgt_, env)),
            ('src', lambda: self.visit(m.src_, env)),
            ('src_shape', lambda: self.visit(m.src_shape_, env))
        ])

    def visit_tuple(self, tup: Tuple, env: None):
        self._write_cls(tup)
        self._write_pos_multi(list(map(lambda f: lambda: self.visit(f, env), tup.fields_)))

    def visit_list(self, lst: List, env: None):
        self._write_cls(lst)
        self._write_named_multi([
            ('len', lambda: self.visit(lst.len_, env)),
            ('idx', lambda: self.visit_symbol(lst.idx_, env)),
            ('body', lambda: self.visit(lst.body_, env))
        ])

    def visit_getitem(self, getitem: GetItem, env: None):
        self.visit(getitem.arr_, env)
        self._write_pos([lambda: self.visit(getitem.idx_, env)], prefix='[', suffix=']')

    def visit_len(self, ln: Len, env: None):
        self._write_cls(ln)
        self._write_pos([lambda: self.visit(ln.arr_, env)])

    def visit_concat(self, concat: Concat, env: None):
        self._write_cls(concat)
        self._write_pos_multi(list(map(lambda arr: lambda: self.visit(arr, env), concat.arrays_)))

    def visit_slice(self, slc: Slice, env: None):
        self.visit(slc.arr_, env)
        self._write_pos([lambda: self.visit(slc.ran_, env)], prefix='[', suffix=']')

    def visit_map(self, m: Map, env: None):
        self._write_cls(m)
        self._write_named_multi([
            ('arr', lambda: self.visit(m.arr_, env)),
            ('sym', lambda: self.visit(m.sym_, env)),
            ('body', lambda: self.visit(m.body_, env))
        ])

    def visit_reduce_array(self, red: ReduceArray, env: None):
        self._write_cls(red)
        self._write_named_multi([
            ('arr', lambda: self.visit(red.arr_, env)),
            ('op', lambda: self._buf.write(red.op_.value)),
            ('init', lambda: self.visit(red.init_, env))
        ])

    def visit_reduce_index(self, red: ReduceRange, env: None):
        self._write_cls(red)
        self._write_named_multi([
            ('ran', lambda: self.visit(red.ran_, env)),
            ('op', lambda: self._buf.write(red.op_.value)),
            ('idx', lambda: self.visit(red.idx_, env)),
            ('body', lambda: self.visit(red.body_, env)),
            ('init', lambda: self.visit(red.init_, env))
        ])

    def visit_filter(self, flt: Filter, env: None):
        self._write_cls(flt)
        self._write_named_multi([
            ('arr', lambda: self.visit(flt.arr_, env)),
            ('sym', lambda: self.visit(flt.sym_, env)),
            ('pred', lambda: self.visit(flt.pred_, env))
        ])

    def visit_inset(self, inset: InSet, env: None):
        self._write_cls(inset)
        self._write_named_multi([
            ('elem', lambda: self.visit(inset.elem_, env)),
            ('set', lambda: self.visit(inset.set_, env))
        ])

    def visit_subset(self, subset: Subset, env: None):
        self._write_cls(subset)
        self._write_named_multi([
            ('sub', lambda: self.visit(subset.sub_, env)),
            ('sup', lambda: self.visit(subset.sup_, env))
        ])

    def visit_perm(self, perm: Perm, env: None):
        self._write_cls(perm)
        self._write_named_multi([
            ('tgt', lambda: self.visit(perm.tgt_, env)),
            ('src', lambda: self.visit(perm.src_, env))
        ])

    def _write_cls(self, e: Expr):
        self._buf.write(cls_name(e))

    def _write_pos(self, items: t.List[Callable[[], None]],
                   sep: str = ', ', prefix: str = '(', suffix: str = ')'):
        self._buf.write_pos(items, sep=sep, prefix=prefix, suffix=suffix)

    def _write_pos_multi(self, items: t.List[Callable[[], None]],
                         sep: str = ',', prefix: str = '(', suffix: str = ')'):
        self._buf.write_pos_multi(items, sep=sep, prefix=prefix, suffix=suffix)

    def _write_named(self, items: t.List[t.Tuple[str, Callable[[], None]]],
                     sep: str = ', ', prefix: str = '(', suffix: str = ')'):
        self._buf.write_named(items, sep=sep, prefix=prefix, suffix=suffix)

    def _write_named_multi(self, items: t.List[t.Tuple[str, Callable[[], None]]],
                           sep: str = ',', prefix: str = '(', suffix: str = ')'):
        self._buf.write_named_multi(items, sep=sep, prefix=prefix, suffix=suffix)


def print_expr(expr: Expr, buf: CodeBuffer, highlights: t.List[Expr]):
    ExprPrinter(buf, highlights).visit(expr, None)
