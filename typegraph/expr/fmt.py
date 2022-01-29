import typing as t
from typing import Any, Callable

from colorama import Fore, Back

from .array import Tuple, List, GetItem, Len, Concat, Slice, Map, ReduceArray, ReduceIndex, Filter, \
    InSet, Subset
from .basic import Expr, Const, Var, Range, Symbol, Env, Arith, Cmp, Not, And, Or, ForAll, Cond, \
    GetAttr, Dummy
from .tensor import Num, TensorDesc, Shape, Rank, GetDType
from .visitor import ExprVisitor
from ..util import CodeBuffer, NameGenerator, Ref, cls_name, colored_text


class ExprPrinter(ExprVisitor[Env[str], Any]):
    def __init__(self, buf: CodeBuffer, highlights: t.List[Expr]):
        super().__init__()
        self._buf = buf
        self._high = set(Ref(e) for e in highlights)
        self._name_gen = NameGenerator('_s', [])
        self._color_idx = 0

    def visit(self, expr: Expr, env: Env[str]):
        high = Ref(expr) in self._high
        if high:
            self._buf.write(Back.RED + Fore.BLACK)
        super().visit(expr, env)
        if high:
            self._buf.write(Back.RESET + Fore.RESET)

    def visit_const(self, const: Const, env: Env[str]):
        self._buf.write(str(const.val_))

    _id_mask = (1 << 16) - 1

    def visit_var(self, var: Var, env: Env[str]):
        self._write_cls(var)
        items: t.List[t.Tuple[str, Callable[[], None]]] = [
            ('id', lambda: self._buf.write(hex(id(var) & self._id_mask))),
            ('tmpl', lambda: self._buf.write(str(var.tmpl_))),
        ]
        if var.type_ is not None:
            items.append(('ty', lambda: self._buf.write(str(var.type_))))
        if var.ran_ is not None:
            items.append(('ran', lambda: self.visit(var.ran_, env)))
        self._write_named(items)

    def visit_range(self, ran: Range, env: Env[str]):
        self._write_cls(ran)
        items: t.List[t.Tuple[str, Callable[[], None]]] = []
        if ran.begin_ is not None:
            items.append(('begin', lambda: self.visit(ran.begin_, env)))
        if ran.end_ is not None:
            items.append(('end', lambda: self.visit(ran.end_, env)))
        self._write_named(items)

    def visit_symbol(self, sym: Symbol, env: Env[str]):
        self._buf.write(f'{cls_name(sym)}(\'{env[sym]}\')')

    colors = [
        Fore.YELLOW,
        Fore.GREEN,
        Fore.BLUE,
    ]

    def _next_color(self):
        color = self.colors[self._color_idx]
        self._color_idx = (self._color_idx + 1) % len(self.colors)
        return color

    def visit_arith(self, arith: Arith, env: Env[str]):
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

    def visit_cmp(self, cmp: Cmp, env: Env[str]):
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

    def visit_not(self, n: Not, env: Env[str]):
        self._write_cls(n)
        self._write_pos([n.prop_, lambda prop: self.visit(prop, env)])

    def visit_and(self, a: And, env: Env[str]):
        self._write_cls(a)
        self._write_pos_multi(list(map(lambda c: lambda: self.visit(c, env), a.clauses_)))

    def visit_or(self, o: Or, env: Env[str]):
        self._write_cls(o)
        self._write_pos_multi(list(map(lambda c: lambda: self.visit(c, env), o.clauses_)))

    def visit_forall(self, forall: ForAll, env: Env[str]):
        nested_env = self._gen_nested_env(env, forall.idx_)
        self._write_cls(forall)
        self._write_named_multi([
            ('ran', lambda: self.visit(forall.ran_, env)),
            ('idx', lambda: self.visit_symbol(forall.idx_, nested_env)),
            ('body', lambda: self.visit(forall.body_, nested_env))
        ])

    def visit_cond(self, cond: Cond, env: Env[str]):
        self._write_cls(cond)
        self._write_named_multi([
            ('pred', lambda: self.visit(cond.pred_, env)),
            ('tr_br', lambda: self.visit(cond.tr_br_, env)),
            ('fls_br', lambda: self.visit(cond.fls_br_, env))
        ])

    def visit_attr(self, attr: GetAttr, env: Env[str]):
        self._buf.write(f'a(\'{attr.name_}\')')

    def visit_dummy(self, dum: Dummy, env: Env[str]):
        self._buf.write(f'{cls_name(dum)}()')

    def visit_num(self, num: Num, env: Env[str]):
        self._buf.write(f'{num.t_kind_.value}.num')

    def visit_shape(self, shape: Shape, env: Env[str]):
        self._write_tensor_attr(shape.tensor_, 'shape', env)

    def visit_rank(self, rank: Rank, env: Env[str]):
        self._write_tensor_attr(rank.tensor_, 'rank', env)

    def visit_dtype(self, dtype: GetDType, env: Env[str]):
        self._write_tensor_attr(dtype.tensor_, 'dtype', env)

    def _write_tensor_attr(self, desc: TensorDesc, name: str, env: Env[str]):
        self._buf.write(f'{desc.kind_.value}[')
        self.visit(desc.idx_, env)
        self._buf.write(f'].{name}')

    def visit_tuple(self, tup: Tuple, env: Env[str]):
        self._write_cls(tup)
        self._write_pos_multi(list(map(lambda f: lambda: self.visit(f, env), tup.fields_)))

    def visit_list(self, lst: List, env: Env[str]):
        nested_env = self._gen_nested_env(env, lst.idx_)
        self._write_cls(lst)
        self._write_named_multi([
            ('len', lambda: self.visit(lst.len_, env)),
            ('idx', lambda: self.visit_symbol(lst.idx_, nested_env)),
            ('body', lambda: self.visit(lst.body_, nested_env))
        ])

    def visit_getitem(self, getitem: GetItem, env: Env[str]):
        self.visit(getitem.arr_, env)
        self._write_pos([lambda: self.visit(getitem.idx_, env)], prefix='[', suffix=']')

    def visit_len(self, ln: Len, env: Env[str]):
        self._write_cls(ln)
        self._write_pos([lambda: self.visit(ln.arr_, env)])

    def visit_concat(self, concat: Concat, env: Env[str]):
        self._write_cls(concat)
        self._write_pos_multi(list(map(lambda arr: lambda: self.visit(arr, env), concat.arrays_)))

    def visit_slice(self, slc: Slice, env: Env[str]):
        self.visit(slc.arr_, env)
        self._write_pos([lambda: self.visit(slc.ran_, env)], prefix='[', suffix=']')

    def visit_map(self, m: Map, env: Env[str]):
        nested_env = self._gen_nested_env(env, m.sym_)
        self._write_cls(m)
        self._write_named_multi([
            ('arr', lambda: self.visit(m.arr_, env)),
            ('sym', lambda: self.visit(m.sym_, nested_env)),
            ('body', lambda: self.visit(m.body_, nested_env))
        ])

    def visit_reduce_array(self, red: ReduceArray, env: Env[str]):
        self._write_cls(red)
        self._write_named_multi([
            ('arr', lambda: self.visit(red.arr_, env)),
            ('op', lambda: self._buf.write(red.op_.value)),
            ('init', lambda: self.visit(red.init_, env))
        ])

    def visit_reduce_index(self, red: ReduceIndex, env: Env[str]):
        nested_env = self._gen_nested_env(env, red.idx_)
        self._write_cls(red)
        self._write_named_multi([
            ('ran', lambda: self.visit(red.ran_, env)),
            ('op', lambda: self._buf.write(red.op_.value)),
            ('idx', lambda: self.visit(red.idx_, nested_env)),
            ('body', lambda: self.visit(red.body_, nested_env)),
            ('init', lambda: self.visit(red.init_, env))
        ])

    def visit_filter(self, flt: Filter, env: Env[str]):
        nested_env = self._gen_nested_env(env, flt.sym_)
        self._write_cls(flt)
        self._write_named_multi([
            ('arr', lambda: self.visit(flt.arr_, env)),
            ('sym', lambda: self.visit(flt.sym_, nested_env)),
            ('pred', lambda: self.visit(flt.pred_, nested_env))
        ])

    def visit_inset(self, inset: InSet, env: Env[str]):
        self._write_cls(inset)
        self._write_named_multi([
            ('elem', lambda: self.visit(inset.elem_, env)),
            ('set', lambda: self.visit(inset.set_, env))
        ])

    def visit_subset(self, subset: Subset, env: Env[str]):
        self._write_cls(subset)
        self._write_named_multi([
            ('sub', lambda: self.visit(subset.sub_, env)),
            ('sup', lambda: self.visit(subset.sup_, env))
        ])

    def _gen_nested_env(self, env: Env[str], sym: Symbol):
        name = self._name_gen.generate()
        return env + (sym, name)

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
    ExprPrinter(buf, highlights).visit(expr, Env())
