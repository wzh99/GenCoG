from typing import Dict, Callable, Set, Iterable

import z3
from numpy.random import Generator

from .store import ValueStore
from ..expr.array import Tuple, List, GetItem, Len, Concat, Slice, Map, ReduceArray, ReduceIndex, \
    Filter, InSet, Subset
from ..expr.basic import Env, Expr, Const, Var, Symbol, Range, Arith, Cmp, Not, And, Or, ForAll, \
    Cond, GetAttr, Dummy, ArithOp
from ..expr.tensor import Num, Shape, Rank, GetDType
from ..expr.ty import Type, ValueType, BOOL, INT, FLOAT
from ..expr.visitor import ExprVisitor
from ..util import NameGenerator, Ref

z3.set_param('smt.phase_selection', 5)

_z3_var_funcs: Dict[Type, Callable[[str], z3.ExprRef]] = {
    BOOL: lambda s: z3.Bool(s),
    INT: lambda s: z3.BitVec(s, 16),
    FLOAT: lambda s: z3.Real(s),
}


def solve_smt(var_set: Set[Ref[Var]], constrs: Iterable[Expr], store: ValueStore,
              rng: Generator) -> bool:
    # Create Z3 variables for all TypeGraph variables
    name_gen = NameGenerator('_v', [])
    var_map = dict((ref, _z3_var_funcs[ref.obj_.type_](name_gen.generate()))
                   for ref in var_set)

    # Generate variable range constraints

    return False


class Z3ExprGen(ExprVisitor[Env[z3.ExprRef], z3.ExprRef]):

    def __init__(self, var_map: Dict[Ref[Var], z3.ExprRef]):
        super().__init__()
        self._var_map = var_map

    val_funcs: Dict[Type, Callable[[ValueType], z3.ExprRef]] = {
        BOOL: lambda v: z3.BoolVal(v),
        INT: lambda v: z3.BitVecVal(v, 16),
        FLOAT: lambda v: z3.RealVal(v)
    }

    def visit_const(self, const: Const, env: Env[z3.ExprRef]) -> z3.ExprRef:
        return self.val_funcs[const.type_](const.val_)

    def visit_var(self, var: Var, env: Env[z3.ExprRef]) -> z3.ExprRef:
        return self._var_map[Ref(var)]

    def visit_symbol(self, sym: Symbol, env: Env[z3.ExprRef]) -> z3.ExprRef:
        return env[sym]

    def visit_range(self, ran: Range, env: Env[z3.ExprRef]) -> z3.ExprRef:
        raise False

    z3_arith_map: Dict[ArithOp, Callable[[z3.ExprRef, z3.ExprRef], z3.ExprRef]] = {
        ArithOp.ADD: z3.ArithRef.__add__,
        ArithOp.SUB: z3.ArithRef.__sub__,
        ArithOp.MUL: z3.ArithRef.__mul__,
        ArithOp.DIV: z3.ArithRef.__truediv__,
        ArithOp.MOD: z3.ArithRef.__mod__,
        ArithOp.MAX: lambda l, r: z3.Cond(l >= r, l, r),
        ArithOp.MIN: lambda l, r: z3.Cond(l <= r, l, r),
    }

    def visit_arith(self, arith: Arith, env: Env[z3.ExprRef]) -> z3.ExprRef:
        return self._visit_sub(arith, env)

    def visit_cmp(self, cmp: Cmp, env: Env[z3.ExprRef]) -> z3.ExprRef:
        return self._visit_sub(cmp, env)

    def visit_not(self, n: Not, env: Env[z3.ExprRef]) -> z3.ExprRef:
        return self._visit_sub(n, env)

    def visit_and(self, a: And, env: Env[z3.ExprRef]) -> z3.ExprRef:
        return self._visit_sub(a, env)

    def visit_or(self, o: Or, env: Env[z3.ExprRef]) -> z3.ExprRef:
        return self._visit_sub(o, env)

    def visit_forall(self, forall: ForAll, env: Env[z3.ExprRef]) -> z3.ExprRef:
        return self._visit_sub(forall, env)

    def visit_cond(self, cond: Cond, env: Env[z3.ExprRef]) -> z3.ExprRef:
        return self._visit_sub(cond, env)

    def visit_attr(self, attr: GetAttr, env: Env[z3.ExprRef]) -> z3.ExprRef:
        return self._visit_sub(attr, env)

    def visit_dummy(self, dum: Dummy, env: Env[z3.ExprRef]) -> z3.ExprRef:
        pass

    def visit_num(self, num: Num, env: Env[z3.ExprRef]) -> z3.ExprRef:
        return self._visit_sub(num, env)

    def visit_shape(self, shape: Shape, env: Env[z3.ExprRef]) -> z3.ExprRef:
        return self._visit_sub(shape, env)

    def visit_rank(self, rank: Rank, env: Env[z3.ExprRef]) -> z3.ExprRef:
        return self._visit_sub(rank, env)

    def visit_dtype(self, dtype: GetDType, env: Env[z3.ExprRef]) -> z3.ExprRef:
        return self._visit_sub(dtype, env)

    def visit_tuple(self, tup: Tuple, env: Env[z3.ExprRef]) -> z3.ExprRef:
        return self._visit_sub(tup, env)

    def visit_list(self, lst: List, env: Env[z3.ExprRef]) -> z3.ExprRef:
        return self._visit_sub(lst, env)

    def visit_getitem(self, getitem: GetItem, env: Env[z3.ExprRef]) -> z3.ExprRef:
        return self._visit_sub(getitem, env)

    def visit_len(self, ln: Len, env: Env[z3.ExprRef]) -> z3.ExprRef:
        return self._visit_sub(ln, env)

    def visit_concat(self, concat: Concat, env: Env[z3.ExprRef]) -> z3.ExprRef:
        return self._visit_sub(concat, env)

    def visit_slice(self, slc: Slice, env: Env[z3.ExprRef]) -> z3.ExprRef:
        return self._visit_sub(slc, env)

    def visit_map(self, m: Map, env: Env[z3.ExprRef]) -> z3.ExprRef:
        return self._visit_sub(m, env)

    def visit_reduce_array(self, red: ReduceArray, env: Env[z3.ExprRef]) -> z3.ExprRef:
        return self._visit_sub(red, env)

    def visit_reduce_index(self, red: ReduceIndex, env: Env[z3.ExprRef]) -> z3.ExprRef:
        return self._visit_sub(red, env)

    def visit_filter(self, flt: Filter, env: Env[z3.ExprRef]) -> z3.ExprRef:
        return self._visit_sub(flt, env)

    def visit_inset(self, inset: InSet, env: Env[z3.ExprRef]) -> z3.ExprRef:
        return self._visit_sub(inset, env)

    def visit_subset(self, subset: Subset, env: Env[z3.ExprRef]) -> z3.ExprRef:
        return self._visit_sub(subset, env)
