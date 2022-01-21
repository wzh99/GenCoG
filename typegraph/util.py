from typing import Callable, TypeVar, Optional, List, cast

T = TypeVar('T')
R = TypeVar('R')


def cls_name(o) -> str:
    return o.__class__.__name__


def map_optional(f: Callable[[T], R], o: Optional[T]) -> Optional[R]:
    return None if o is None else f(o)


def unwrap(o: Optional[T]) -> T:
    if o is None:
        raise RuntimeError('Cannot unwrap None.')
    else:
        return cast(T, o)


def filter_none(lst: List[Optional[T]]) -> List[T]:
    return list(filter(lambda e: e is not None, lst))
