from typing import Iterable, TypeVar, Generic, Callable

from typefuzz import Op
from typefuzz.graph.base import Value
from typefuzz.solve import TensorType
from typefuzz.spec import max_rank, common_dtypes
from typefuzz.util import StaticBitMap, Ref, DynamicBitMap, BitMap

T = TypeVar('T')
K = TypeVar('K')


class OpLookup:
    """
    Lookup operators with combination of properties.
    """

    def __init__(self, ops: Iterable[Op]):
        # Create bitmap
        ops = list(ops)
        self._bit_map = StaticBitMap(Ref(op) for op in ops)

        # Create table from different properties
        op_specs = dict((op, op.spec) for op in ops)
        self._first_ranks = StaticSetTable(range(2, max_rank + 1), ops, self._bit_map,
                                           lambda op: op_specs[op].first_rank_choices)
        self._first_dtypes = StaticSetTable(common_dtypes, ops, self._bit_map,
                                            lambda op: op_specs[op].first_dtype_choices)

    def by_first_tensor(self, ty: TensorType) -> Iterable[Op]:
        a = self._first_ranks[ty.rank] & self._first_dtypes[ty.dtype_]
        return map(Ref.get, self._bit_map.decode(a))


class ValueLookup:
    """
    Lookup tensor values with type patterns.
    """

    def __init__(self):
        self._bit_map: DynamicBitMap[Ref[Value]] = DynamicBitMap()
        self._ranks = DynamicSetTable(range(2, max_rank + 1), self._bit_map)
        self._dtypes = DynamicSetTable(common_dtypes, self._bit_map)

    def add(self, value: Value):
        self._ranks.add(value, [value.type_.rank])
        self._dtypes.add(value, [value.type_.dtype_])


class SetTable(Generic[T, K]):
    """
    Mapping from keys to set of objects.
    """

    def __init__(self, all_keys: Iterable[K], bit_map: BitMap[Ref[T]]):
        self._table = dict((k, bit_map.empty) for k in all_keys)

    def __getitem__(self, k: K):
        self._check_key(k)
        return self._table[k]

    def _check_key(self, k: K):
        if k not in self._table:
            raise ValueError(f'Undefined key {k}.')


class StaticSetTable(SetTable[T, K]):
    """
    Static set table, where objects are known before construction.
    """

    def __init__(self, all_keys: Iterable[K], objs: Iterable[T], bit_map: StaticBitMap[Ref[T]],
                 keys_f: Callable[[T], Iterable[K]]):
        super().__init__(all_keys, bit_map)
        self._bit_map = bit_map
        for o in objs:
            for k in keys_f(o):
                if k in self._table:
                    bit_map.set(self._table[k], Ref(o))


class DynamicSetTable(SetTable[T, K]):
    """
    Dynamic set table, where objects are dynamically added to the table.
    """

    def __init__(self, all_keys: Iterable[K], bit_map: DynamicBitMap[Ref[T]]):
        super().__init__(all_keys, bit_map)
        self._bit_map = bit_map

    def add(self, obj: T, keys: Iterable[K]):
        # Extend bitmap if the object is not in universal set
        ref = Ref(obj)
        if ref not in self._bit_map:
            self._bit_map.add(ref)

        # Synchronize bit vectors with bitmap
        for a in self._table.values():
            a.extend([0] * (len(self._bit_map) - len(a)))

        # Set corresponding bit in each bit vector
        key_set = set(keys)
        for k, a in self._table.items():
            self._bit_map.set(a, ref, b=k in key_set)
