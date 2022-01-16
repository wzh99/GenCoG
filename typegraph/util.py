from typing import Callable, TypeVar, Optional, cast


def cls_name(o) -> str:
    return o.__class__.__name__


T = TypeVar('T')
R = TypeVar('R')


def map_optional(f: Callable[[T], R], o: Optional[T]) -> Optional[R]:
    return None if o is None else f(o)


def unwrap(o: Optional[T]) -> T:
    if o is None:
        raise RuntimeError('Cannot unwrap \'None\'.')
    else:
        return cast(T, o)
