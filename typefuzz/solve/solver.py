import typing as t
from typing import Dict, List, cast

from numpy.random import Generator

from .eval import PartialEval, EvalExpr, EvalError
from .smt import solve_smt
from .store import ValueStore, StoreNode, NodeKind, ValueStatus, ScalarNode, ArrayNode, StoreError
from .valid import validate
from ..expr.array import Tuple
from ..expr.basic import ExprKind, Const, And, Var, Cmp, CmpOp
from ..expr.fmt import print_expr
from ..expr.ty import DataType, ValueType, BOOL, INT, FLOAT
from ..expr.visitor import CopyExpr, StructuralEq
from ..spec import TypeSpec
from ..util import CodeBuffer, Ref, cls_name


class TensorType:
    def __init__(self, shape: List[int], dtype: DataType):
        self.shape_ = shape
        self.dtype_ = dtype

    @property
    def rank(self):
        return len(self.shape_)

    def __eq__(self, other: 'TensorType'):
        return self.shape_ == other.shape_ and self.dtype_ == other.dtype_

    def __repr__(self):
        # Compatible with Relay tensor type
        return f'Tensor[{tuple(self.shape_)}, {self.dtype_}]'


class OpTypeInfo:
    def __init__(self, attrs: List[t.Tuple[str, ValueType]], in_types: List[TensorType],
                 out_types: List[TensorType]):
        self.attrs_ = attrs
        self.in_types_ = in_types
        self.out_types_ = out_types

    def __repr__(self):
        buf = CodeBuffer()
        buf.write(cls_name(self))
        buf.write_named_multi([
            ('attrs', lambda: buf.write_named_multi(
                map(lambda p: (p[0], lambda: buf.write(str(p[1]))), self.attrs_),
                prefix='[', suffix=']'
            )),
            ('in_types', lambda: buf.write_pos_multi(
                map(lambda tt: lambda: buf.write(str(tt)), self.in_types_),
                prefix='[', suffix=']'
            )),
            ('out_types', lambda: buf.write_pos_multi(
                map(lambda tt: lambda: buf.write(str(tt)), self.out_types_),
                prefix='[', suffix=']'
            ))
        ])
        return str(buf)


class SolveError(Exception):
    """
    The solver cannot solve type constraints.
    """

    def __init__(self, solver: 'TypeSolver', msg: str):
        self.solver_ = solver
        self.msg_ = msg

    def __str__(self):
        return f'{self.msg_}\n' \
               f'{str(self.solver_)}'


class TypeSolver:
    """
    Solver of type constraints.
    """

    def __init__(self, spec: TypeSpec, known: Dict[int, TensorType], rng: Generator):
        # Initialize value store and expression visitors
        self._spec = spec
        self.store_ = ValueStore(spec.attrs)
        self._known = known
        self._rng = rng
        self._partial = PartialEval(self.store_, rng)
        self._eval = EvalExpr(self.store_)
        self._eq = StructuralEq()

        # Copy expressions to avoid polluting original specification
        cp = CopyExpr()
        self._in_ranks = cp.copy(spec.in_ranks)
        self._extra = [cp.copy(e) for e in spec.extra]
        shape_root = self.store_.in_shapes_
        in_num = cp.copy(spec.in_num)
        shape_root.set_len_defined(in_num)
        shape_root.set_expr_defined(cp.copy(spec.in_shapes))
        dtype_root = self.store_.in_dtypes_
        dtype_root.set_len_defined(in_num)
        dtype_root.set_expr_defined(cp.copy(spec.in_dtypes))

    def solve(self):
        """
        Perform complete solving of the type constraints.
        """
        # Solve attributes and inputs
        while True:
            while self._solve_one_iter():
                pass
            if not self._solve_smt():
                break

        # Extract solved values from store
        attrs, in_types = self._extract_solved()

        # Evaluate output types
        try:
            out_types = self._eval_out()
        except EvalError as err:
            raise SolveError(self, err.msg_)

        return OpTypeInfo(attrs, in_types, out_types)

    def solve_initial(self):
        """
        Only try initial solving without sampling or SMT solving. This method is mainly used to
        provide hint for computation graph generation.
        """
        while self._solve_one_iter():
            pass

    def _solve_one_iter(self):
        # Solve attributes
        changed = False
        try:
            for _, node in self.store_.attrs_:
                changed |= self._solve_node(node)

            # Solve inputs
            changed |= self._solve_shapes(self.store_.in_shapes_)
            changed |= self._solve_dtypes(self.store_.in_dtypes_)

            # Solve extra constraints
            changed |= self._solve_extra()
        except StoreError as err:
            raise SolveError(self, err.msg_)

        return changed

    def _solve_shapes(self, root: ArrayNode) -> bool:
        # Solve number
        changed = False
        if not root.len_solved:
            changed |= self._solve_len(root, by_elem=False)
            if not root.len_solved:
                return changed

        # Solve tensors
        if not root.elem_defined:
            # Partially evaluate ranks
            ranks = self._partial.transform(self._in_ranks)
            if ranks.kind != ExprKind.TUPLE:
                return changed
            ranks = cast(Tuple, ranks)
            if root.len_.value != len(ranks.fields_):
                raise SolveError(
                    self,
                    f'Length of input rank array {len(ranks.fields_)} is not consistent with '
                    f'input number {root.len_.value}. '
                )

            # Define ranks for each input tensor
            for tensor, rank in zip(root.children_, ranks.fields_):
                tensor = cast(ArrayNode, tensor)
                tensor.set_len_defined(rank)

            # Partially evaluate shapes
            shapes = self._partial.transform(root.expr_)
            if shapes.kind != ExprKind.TUPLE:
                return changed
            shapes = cast(Tuple, shapes)
            if root.len_.value != len(shapes.fields_):
                raise SolveError(
                    self,
                    f'Length of input shape array {len(shapes.fields_)} is not consistent with '
                    f'input number {root.len_.value}. '
                )

            # Define shapes for each input tensor
            root.set_elem_defined(shapes)
            changed = True

        # Solve shapes
        for t_idx, tensor in enumerate(root.children_):
            # Solve rank
            tensor = cast(ArrayNode, tensor)
            prev_solved = tensor.len_solved
            changed |= self._solve_len(tensor, by_elem=False)
            if t_idx in self._known:
                known_rank = self._known[t_idx].rank
                self._try_add_extra(tensor.len_, known_rank)
                tensor.set_len_solved(known_rank)
            changed |= prev_solved != tensor.len_solved
            if not tensor.len_solved:
                continue

            # Partially evaluate dimensions
            if not tensor.elem_defined:
                shape = self._partial.transform(tensor.expr_)
                if shape.kind != ExprKind.TUPLE:
                    continue
                shape = cast(Tuple, shape)
                if tensor.len_.value != len(shape.fields_):
                    raise SolveError(
                        self,
                        f'Length of input shape {len(shape.fields_)} for tensor {t_idx} is not '
                        f'consistent with rank {tensor.len_.value}. '
                    )
                tensor.set_elem_defined(shape)
                changed = True

            # Solve dimensions
            for d_idx, dim in enumerate(tensor.children_):
                dim = cast(ScalarNode, dim)
                prev_solved = dim.solved
                changed |= self._solve_scalar(dim)
                if t_idx in self._known:
                    known_dim = self._known[t_idx].shape_[d_idx]
                    self._try_add_extra(dim, known_dim)
                    dim.set_solved(known_dim)
                changed |= prev_solved != dim.solved

        return changed

    def _solve_dtypes(self, root: ArrayNode) -> bool:
        # Solve number
        changed = False
        if not root.len_solved:
            changed |= self._solve_len(root, by_elem=False)
            if not root.len_solved:
                return changed

        # Solve tensors
        if not root.elem_defined:
            # Partially evaluate data types
            dtypes = self._partial.transform(root.expr_)
            if dtypes.kind != ExprKind.TUPLE:
                return changed
            dtypes = cast(Tuple, dtypes)
            if root.len_.value != len(dtypes.fields_):
                raise SolveError(
                    self,
                    f'Length of input rank array {len(dtypes.fields_)} is not consistent with '
                    f'input number {root.len_.value}. '
                )

            # Define data type for each tensor
            root.set_elem_defined(dtypes)
            changed = True

        # Solve data types
        for t_idx, dtype in enumerate(root.children_):
            dtype = cast(ScalarNode, dtype)
            prev_solved = dtype.solved
            if t_idx in self._known:
                dtype.set_solved(self._known[t_idx].dtype_)
            changed |= self._solve_scalar(dtype)
            changed |= prev_solved != dtype.solved

        return changed

    def _solve_extra(self) -> bool:
        new_extra = []
        changed = False

        for e in self._extra:
            post = self._partial.transform(e)
            if post.kind == ExprKind.CONST:
                const = cast(Const, post)
                if const.val_ is False:
                    raise SolveError(
                        self, 'Extra constraint is not satisfiable.'
                    )
                changed = True
                continue
            elif post.kind == ExprKind.AND:
                and_e = cast(And, post)
                new_extra.extend(and_e.clauses_)
                changed = True
                continue
            elif post.kind == ExprKind.CMP:
                cmp = cast(Cmp, post)
                if cmp.op_ == CmpOp.EQ and cmp.lhs_.kind == ExprKind.VAR and \
                        cmp.rhs_.kind == ExprKind.CONST:
                    lhs = cast(Var, cmp.lhs_)
                    rhs = cast(Const, cmp.rhs_)
                    self.store_.set_var_solved(lhs, rhs.val_)
                    changed = True
                    continue

            changed |= not self._eq.visit(e, post)
            new_extra.append(post)

        self._extra = new_extra
        return changed

    def _solve_smt(self):
        # Find all valid variables and constraints with union-find
        changed = False
        union = validate(self.store_, self._extra)
        all_valid = list(union.all_valid())
        if len(all_valid) == 0:
            return False

        # Process valid expressions
        all_vars = []
        extra = []
        for e in all_valid:
            # Differentiate variables and constraints
            if e.kind != ExprKind.VAR:
                extra.append(e)
                continue
            var = cast(Var, e)
            if Ref(var) in all_vars:
                continue

            # Try sample variables only bounded by its own range
            if union.has_use(var):
                all_vars.append(Ref(var))
            elif self._try_sample(var):
                changed = True
            else:
                raise SolveError(
                    self, 'Cannot solve unconstrained variable.'
                )

        # Solve by SMT
        changed |= solve_smt(all_vars, extra, self.store_, self._rng)
        return changed

    def _try_sample(self, var: Var):
        # Directly sample boolean values
        if var.type_ == BOOL:
            v = bool(self._rng.integers(2))
            self.store_.set_var_solved(var, v)
            return True

        # Sample variables if choices are provided
        if var.choices_ is not None:
            try:
                choices = tuple(self._eval.evaluate(var.choices_))
                v = choices[self._rng.choice(len(choices))]
                self.store_.set_var_solved(var, v)
                return True
            except EvalError:
                return False

        # Get range for numeric values
        if var.ran_ is not None:
            ran = var.ran_
            if ran.begin_ is None or ran.begin_.kind != ExprKind.CONST:
                return False
            low = cast(Const, ran.begin_).val_
            if ran.end_ is None or ran.end_.kind != ExprKind.CONST:
                return False
            high = cast(Const, ran.end_).val_
            if low >= high:
                return False

            # Sample numeric values
            if var.type_ == INT:
                v = int(self._rng.integers(low=low, high=high))
                self.store_.set_var_solved(var, v)
                return True
            elif var.type_ == FLOAT:
                v = float(self._rng.uniform(low=low, high=high))
                self.store_.set_var_solved(var, v)
                return True
            else:
                return False

        return False

    def _extract_solved(self):
        # Extract attributes
        attrs = []
        for name, node in self.store_.attrs_:
            if not node.solved:
                raise SolveError(
                    self, f'Attribute \'{name}\' not solved.'
                )
            attrs.append((name, cast(ValueType, node.value)))

        # Extract input types
        if not self.store_.in_shapes_.solved:
            raise SolveError(
                self, 'Input shapes not solved.'
            )
        in_shapes = cast(List[List[int]], self.store_.in_shapes_.value)
        if not self.store_.in_dtypes_.solved:
            raise SolveError(
                self, 'Input data types not solved.'
            )
        in_dtypes = cast(List[DataType], self.store_.in_dtypes_.value)
        assert len(in_shapes) == len(in_dtypes)
        in_types = [TensorType(shape, dtype) for shape, dtype in zip(in_shapes, in_dtypes)]

        return attrs, in_types

    def _eval_out(self):
        # Evaluate output number
        num_expr = self._spec.out_num
        num = self._eval.evaluate(num_expr)
        shapes_node = self.store_.out_shapes_
        dtypes_node = self.store_.out_dtypes_
        shapes_node.set_expr_defined(self._spec.out_shapes)
        shapes_node.set_len_solved(num)
        dtypes_node.set_expr_defined(self._spec.out_dtypes)
        dtypes_node.set_len_solved(num)

        # Evaluate output ranks
        ranks: List[int] = list(self._eval.evaluate(self._spec.out_ranks))
        if len(ranks) != num:
            raise SolveError(
                self,
                f'Length of rank array {len(ranks)} is not consistent with input number {num}.'
            )
        for shape_node, rank in zip(shapes_node.children_, ranks):
            cast(ArrayNode, shape_node).set_len_solved(rank)

        # Evaluate output shapes
        shapes_iter = self._eval.evaluate(self._spec.out_shapes)
        shapes: List[List[int]] = [list(shape) for shape in shapes_iter]
        if len(shapes) != num:
            raise SolveError(
                self,
                f'Length of shape array {len(shapes)} is not consistent with input number {num}.'
            )
        for t_idx, shape in enumerate(shapes):
            if len(shape) != ranks[t_idx]:
                raise SolveError(
                    self,
                    f'Shape length {len(shape)} of tensor {t_idx} is not consistent with rank '
                    f'{ranks[t_idx]}.'
                )
            shape_node = cast(ArrayNode, shapes_node.children_[t_idx])
            for d_idx, dim in enumerate(shape):
                dim_node = cast(ScalarNode, shape_node.children_[d_idx])
                dim_node.set_solved(dim)

        # Evaluate output data types
        dtypes: List[DataType] = list(self._eval.evaluate(self._spec.out_dtypes))
        if len(dtypes) != num:
            raise SolveError(
                self,
                f'Length of data type array {len(dtypes)} is not consistent with input number '
                f'{num}.'
            )
        for dtype_node, dtype in zip(dtypes_node.children_, dtypes):
            cast(ScalarNode, dtype_node).set_solved(dtype)

        # Create output types
        out_types = [TensorType(shape, dtype) for shape, dtype in zip(shapes, dtypes)]
        return out_types

    def _try_add_extra(self, node: ScalarNode, val: ValueType):
        if node.status_ != ValueStatus.DEFINED:
            return
        if node.expr_.kind in (ExprKind.CONST, ExprKind.VAR):
            return
        self._extra.append(Cmp(CmpOp.EQ, node.expr_, Const(val)))

    def _solve_node(self, node: StoreNode) -> bool:
        if node.kind == NodeKind.SCALAR:
            return self._solve_scalar(cast(ScalarNode, node))
        else:
            return self._solve_array(cast(ArrayNode, node))

    def _solve_scalar(self, node: ScalarNode) -> bool:
        if node.status_ == ValueStatus.UNDEFINED:
            return False  # undefined or solved node cannot be processed
        post = self._partial.transform(node.expr_)
        if post.kind == ExprKind.CONST:
            prev_solved = node.solved
            node.set_solved(cast(Const, post).val_)
            return not prev_solved
        else:
            pre = node.expr_
            node.expr_ = post
            return not self._eq.visit(pre, post)

    def _solve_array(self, node: ArrayNode) -> bool:
        # Solve array length
        changed = False
        if not node.len_solved:  # solve length first
            if not self._solve_len(node):
                return False
            changed = True  # length is solved this time

        # Solve element expression
        if node.expr_defined and not node.elem_defined:
            changed |= self._solve_arr_elem(node)

        # Recursively solve children nodes
        for child in node.children_:
            changed |= self._solve_node(child)

        return changed

    def _solve_len(self, node: ArrayNode, by_elem: bool = True) -> bool:
        # Solve length directly
        if node.len_defined and self._solve_scalar(node.len_):
            node.set_len_solved(node.len_.value)
            return True

        # Solve length through partial evaluation of array expression
        if node.expr_defined and by_elem:
            return self._solve_arr_elem(node)

        return False

    def _solve_arr_elem(self, node: ArrayNode) -> bool:
        tup = self._partial.transform(node.expr_)
        if tup.kind != ExprKind.TUPLE:
            return False
        node.set_elem_defined(cast(Tuple, tup))
        return True

    def __repr__(self):
        buf = CodeBuffer()
        buf.writeln('==== SOLVER DUMP BEGIN ====')
        buf.writeln('---- Value Store ----')
        self.store_.print(buf)
        buf.writeln()
        buf.writeln('---- Extra Constraints ----')
        buf.write_pos_multi(
            map(lambda e: lambda: print_expr(e, buf, []), self._extra),
            prefix='[', suffix=']'
        )
        buf.writeln()
        buf.writeln('==== SOLVER DUMP END ====')
        return str(buf)
