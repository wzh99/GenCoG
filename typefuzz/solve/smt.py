import typing as t
from functools import reduce
from typing import Dict, Callable, Iterable, Union, cast

import z3
from numpy.random import Generator

from .store import ValueStore
from ..config import params
from ..expr.array import Tuple, ReduceArray, ReduceRange
from ..expr.basic import Env, Expr, ExprKind, Const, Var, Symbol, Range, Arith, Cmp, Not, And, Or, \
    ArithOp, CmpOp
from ..expr.ty import Type, ValueType, BOOL, INT
from ..expr.visitor import ExprVisitor
from ..util import NameGenerator, Ref

z3.set_param('smt.phase_selection', 5)
bit_vec_len = params['solver.bit_vec_len']


class Z3SolveError(Exception):
    pass


_z3_var_funcs: Dict[Type, Callable[[str], z3.ExprRef]] = {
    BOOL: lambda s: z3.Bool(s),
    INT: lambda s: z3.BitVec(s, bit_vec_len),
}

_z3_extract_funcs: Dict[
    Type, Callable[[Union[z3.BoolRef, z3.BitVecNumRef, z3.RatNumRef]], ValueType]] = {
    BOOL: z3.BoolRef.__bool__,
    INT: z3.BitVecNumRef.as_long,
}


def solve_smt(var_set: Iterable[Ref[Var]], extra: Iterable[Expr], store: ValueStore,
              rng: Generator) -> bool:
    # Create Z3 variables
    name_gen = NameGenerator('_x')
    var_map = list((ref, _z3_var_funcs[ref.obj_.type_](name_gen.generate()))
                   for ref in var_set)

    # Generate variable range/choice constraints
    solver = z3.Solver()
    expr_gen = Z3ExprGen(var_map)
    for ref, z3_var in var_map:
        var = ref.obj_
        if var.ran_ is not None:
            ran = var.ran_
            if ran.begin_ is not None:
                solver.add(z3_var >= expr_gen.generate(ran.begin_))
            if ran.end_ is not None:
                solver.add(z3_var < expr_gen.generate(ran.end_))
        elif var.choices_ is not None:
            choices = cast(Tuple, var.choices_)
            solver.add(z3.Or(*(z3_var == expr_gen.generate(c) for c in choices.fields_)))

    # Generate extra constraints
    for e in extra:
        solver.add(expr_gen.generate(e))

    # Solve constraints multiple times
    z3.set_param('smt.random_seed', rng.integers(1024))
    cand_models = []
    for _ in range(params['solver.max_model_cand']):
        if solver.check() != z3.sat:
            break
        model = solver.model()
        if len(model) == 0:
            break
        cand_models.append(model)
        exclude = z3.Or(*(z3_var != model[z3_var] for _, z3_var in var_map))
        solver.add(exclude)

    # Choose one possible model
    if len(cand_models) == 0:
        return False
    model = cand_models[rng.choice(len(cand_models))]

    # Save results to value store
    for ref, z3_var in var_map:
        var = ref.obj_
        # noinspection PyTypeChecker
        result = model[z3_var]
        if result is None:
            raise Z3SolveError()
        store.set_var_solved(var, _z3_extract_funcs[var.type_](result))

    return True


class Z3ExprGen(ExprVisitor[Env[z3.ExprRef], z3.ExprRef]):

    def __init__(self, var_map: Iterable[t.Tuple[Ref[Var], z3.ExprRef]]):
        super().__init__()
        self._var_map = dict(var_map)

    def generate(self, e: Expr):
        return self.visit(e, Env())

    val_funcs: Dict[Type, Callable[[ValueType], z3.ExprRef]] = {
        BOOL: lambda v: z3.BoolVal(v),
        INT: lambda v: z3.BitVecVal(v, bit_vec_len),
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
        },
        CmpOp.NE: {
            BOOL: z3.BoolRef.__ne__,
            INT: z3.BitVecRef.__ne__,
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

    def visit_reduce_array(self, red: ReduceArray, env: Env[z3.ExprRef]) -> z3.ExprRef:
        if red.arr_.kind != ExprKind.TUPLE:
            raise Z3SolveError()
        arr = cast(Tuple, red.arr_)
        func = self.z3_arith_funcs[red.op_][red.type_]
        return reduce(lambda acc, e: func(acc, self.visit(e, env)), arr.fields_,
                      self.visit(red.init_, env))

    def visit_reduce_index(self, red: ReduceRange, env: Env[z3.ExprRef]) -> z3.ExprRef:
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
