from functools import reduce
from typing import Dict, Callable, Set, Iterable, Union, cast

import z3
from numpy.random import Generator

from .store import ValueStore
from ..config import config
from ..expr.array import Tuple, List, GetItem, Len, Concat, Slice, Map, ReduceArray, ReduceIndex, \
    Filter, InSet, Subset
from ..expr.basic import Env, Expr, ExprKind, Const, Var, Symbol, Range, Arith, Cmp, Not, And, Or, \
    ForAll, Cond, GetAttr, Dummy, ArithOp, CmpOp
from ..expr.tensor import Num, Shape, Rank, GetDType
from ..expr.ty import Type, ValueType, BOOL, INT, FLOAT, STR
from ..expr.visitor import ExprVisitor
from ..util import NameGenerator, Ref

z3.set_param('smt.phase_selection', 5)


class Z3SolveError(Exception):
    pass


_z3_var_funcs: Dict[Type, Callable[[str], z3.ExprRef]] = {
    BOOL: lambda s: z3.Bool(s),
    INT: lambda s: z3.BitVec(s, 16),
    FLOAT: lambda s: z3.Real(s),
    STR: lambda s: z3.String(s),
}

_z3_extract_funcs: Dict[
    Type, Callable[[Union[z3.BoolRef, z3.BitVecNumRef, z3.RatNumRef]], ValueType]] = {
    BOOL: z3.BoolRef.__bool__,
    INT: z3.BitVecNumRef.as_long,
    FLOAT: lambda v: float(z3.RatNumRef.as_decimal(v, 5)),
}


def solve_smt(var_set: Set[Ref[Var]], extra: Iterable[Expr], store: ValueStore,
              rng: Generator) -> bool:
    # Create Z3 variables for all TypeGraph variables
    name_gen = NameGenerator('_v', [])
    var_map = dict((ref, _z3_var_funcs[ref.obj_.type_](name_gen.generate()))
                   for ref in var_set)

    # Generate variable range constraints
    solver = z3.Solver()
    expr_gen = Z3ExprGen(var_map)
    for ref, z3_var in var_map.items():
        var = ref.obj_
        if var.ran_ is None:
            continue
        ran = var.ran_
        if ran.begin_ is not None:
            solver.add(z3_var >= expr_gen.generate(ran.begin_))
        if ran.end_ is not None:
            solver.add(z3_var < expr_gen.generate(ran.end_))

    # Generate extra constraints
    for e in extra:
        solver.add(expr_gen.generate(e))

    # Solve constraints multiple times
    z3.set_param('smt.random_seed', rng.integers(1024))
    cand_models = []
    for _ in range(config['solver.max_model_cand']):
        if solver.check() != z3.sat:
            break
        model = solver.model()
        cand_models.append(model)
        exclude = z3.Or(*(z3_var != model[z3_var] for z3_var in var_map.values()))
        solver.add(exclude)

    # Choose one possible model
    model = cand_models[rng.choice(len(cand_models))]

    # Save results to value store
    for ref, z3_var in var_map.items():
        var = ref.obj_
        # noinspection PyTypeChecker
        if model[z3_var] is None:
            raise Z3SolveError()
        result = model.eval(z3_var)
        store.set_var_solved(var, _z3_extract_funcs[var.type_](result))

    return True


class Z3ExprGen(ExprVisitor[Env[z3.ExprRef], z3.ExprRef]):

    def __init__(self, var_map: Dict[Ref[Var], z3.ExprRef]):
        super().__init__()
        self._var_map = var_map

    def generate(self, e: Expr):
        return self.visit(e, Env())

    val_funcs: Dict[Type, Callable[[ValueType], z3.ExprRef]] = {
        BOOL: lambda v: z3.BoolVal(v),
        INT: lambda v: z3.BitVecVal(v, 16),
        FLOAT: lambda v: z3.RealVal(v),
        STR: lambda v: z3.StringVal(v),
    }

    def visit_const(self, const: Const, env: Env[z3.ExprRef]) -> z3.ExprRef:
        return self.val_funcs[const.type_](const.val_)

    def visit_var(self, var: Var, env: Env[z3.ExprRef]) -> z3.ExprRef:
        return self._var_map[Ref(var)]

    def visit_symbol(self, sym: Symbol, env: Env[z3.ExprRef]) -> z3.ExprRef:
        return env[sym]

    def visit_range(self, ran: Range, env: Env[z3.ExprRef]) -> z3.ExprRef:
        raise False

    z3_arith_funcs: Dict[ArithOp, Dict[Type, Callable[[z3.ExprRef, z3.ExprRef], z3.ExprRef]]] = {
        ArithOp.ADD: {
            INT: z3.BitVecRef.__add__,
        },
        ArithOp.SUB: {
            INT: z3.BitVecRef.__sub__,
        },
        ArithOp.MUL: {
            INT: z3.BitVecRef.__mul__,
        },
        ArithOp.DIV: {
            INT: z3.BitVecRef.__truediv__,
        },
        ArithOp.MOD: {
            INT: z3.BitVecRef.__mod__,
        },
        ArithOp.MAX: {
            INT: lambda l, r: z3.Cond(l >= r, l, r),
        },
        ArithOp.MIN: {
            INT: lambda l, r: z3.Cond(l <= r, l, r),
        }
    }

    def visit_arith(self, arith: Arith, env: Env[z3.ExprRef]) -> z3.ExprRef:
        func = self.z3_arith_funcs[arith.op_][arith.type_]
        return func(self.visit(arith.lhs_, env), self.visit(arith.rhs_, env))

    z3_cmp_funcs: Dict[CmpOp, Dict[Type, Callable[[z3.ExprRef, z3.ExprRef], z3.ExprRef]]] = {
        CmpOp.EQ: {
            BOOL: z3.BoolRef.__eq__,
            INT: z3.BitVecRef.__eq__,
            STR: z3.SeqRef.__eq__,
        },
        CmpOp.NE: {
            BOOL: z3.BoolRef.__ne__,
            INT: z3.BitVecRef.__ne__,
            STR: z3.SeqRef.__ne__,
        },
        CmpOp.LT: {
            INT: z3.BitVecRef.__lt__,
        },
        CmpOp.LE: {
            INT: z3.BitVecRef.__le__,
        },
        CmpOp.GT: {
            INT: z3.BitVecRef.__gt__,
        },
        CmpOp.GE: {
            INT: z3.BitVecRef.__ge__,
        },
    }

    def visit_cmp(self, cmp: Cmp, env: Env[z3.ExprRef]) -> z3.ExprRef:
        func = self.z3_cmp_funcs[cmp.op_][cmp.lhs_.type_]
        return func(self.visit(cmp.lhs_, env), self.visit(cmp.rhs_, env))

    def visit_not(self, n: Not, env: Env[z3.ExprRef]) -> z3.ExprRef:
        return z3.Not(self.visit(n.prop_, env))

    def visit_and(self, a: And, env: Env[z3.ExprRef]) -> z3.ExprRef:
        return z3.And(*(self.visit(c, env) for c in a.clauses_))

    def visit_or(self, o: Or, env: Env[z3.ExprRef]) -> z3.ExprRef:
        return z3.Or(*(self.visit(c, env) for c in o.clauses_))

    def visit_forall(self, forall: ForAll, env: Env[z3.ExprRef]) -> z3.ExprRef:
        raise Z3SolveError()

    def visit_cond(self, cond: Cond, env: Env[z3.ExprRef]) -> z3.ExprRef:
        return z3.Cond(self.visit(cond.pred_, env), self.visit(cond.tr_br_, env),
                       self.visit(cond.fls_br_, env))

    def visit_attr(self, attr: GetAttr, env: Env[z3.ExprRef]) -> z3.ExprRef:
        raise Z3SolveError()

    def visit_dummy(self, dum: Dummy, env: Env[z3.ExprRef]) -> z3.ExprRef:
        raise Z3SolveError()

    def visit_num(self, num: Num, env: Env[z3.ExprRef]) -> z3.ExprRef:
        raise Z3SolveError()

    def visit_shape(self, shape: Shape, env: Env[z3.ExprRef]) -> z3.ExprRef:
        raise Z3SolveError()

    def visit_rank(self, rank: Rank, env: Env[z3.ExprRef]) -> z3.ExprRef:
        raise Z3SolveError()

    def visit_dtype(self, dtype: GetDType, env: Env[z3.ExprRef]) -> z3.ExprRef:
        raise Z3SolveError()

    def visit_tuple(self, tup: Tuple, env: Env[z3.ExprRef]) -> z3.ExprRef:
        raise Z3SolveError()

    def visit_list(self, lst: List, env: Env[z3.ExprRef]) -> z3.ExprRef:
        raise Z3SolveError()

    def visit_getitem(self, getitem: GetItem, env: Env[z3.ExprRef]) -> z3.ExprRef:
        raise Z3SolveError()

    def visit_len(self, ln: Len, env: Env[z3.ExprRef]) -> z3.ExprRef:
        raise Z3SolveError()

    def visit_concat(self, concat: Concat, env: Env[z3.ExprRef]) -> z3.ExprRef:
        raise Z3SolveError()

    def visit_slice(self, slc: Slice, env: Env[z3.ExprRef]) -> z3.ExprRef:
        raise Z3SolveError()

    def visit_map(self, m: Map, env: Env[z3.ExprRef]) -> z3.ExprRef:
        raise Z3SolveError()

    def visit_reduce_array(self, red: ReduceArray, env: Env[z3.ExprRef]) -> z3.ExprRef:
        if red.arr_.kind != ExprKind.TUPLE:
            raise Z3SolveError()
        arr = cast(Tuple, red.arr_)
        func = self.z3_arith_funcs[red.op_][red.type_]
        return reduce(lambda acc, e: func(acc, self.visit(e, env)), arr.fields_,
                      self.visit(red.init_, env))

    def visit_reduce_index(self, red: ReduceIndex, env: Env[z3.ExprRef]) -> z3.ExprRef:
        ran = red.ran_
        if ran.begin_.kind != ExprKind.CONST or ran.end_.kind != ExprKind.CONST:
            raise Z3SolveError()
        begin = cast(Const, ran.begin_).val_
        end = cast(Const, ran.end_).val_
        func = self.z3_arith_funcs[red.op_][red.type_]
        return reduce(
            lambda acc, idx: func(
                acc, self.visit(red.body_, env + (red.idx_, self.visit_const(Const(idx), env)))
            ),
            range(begin, end), self.visit(red.init_, env)
        )

    def visit_filter(self, flt: Filter, env: Env[z3.ExprRef]) -> z3.ExprRef:
        raise Z3SolveError()

    def visit_inset(self, inset: InSet, env: Env[z3.ExprRef]) -> z3.ExprRef:
        raise Z3SolveError()

    def visit_subset(self, subset: Subset, env: Env[z3.ExprRef]) -> z3.ExprRef:
        raise Z3SolveError()
