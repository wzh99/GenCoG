from io import StringIO
from typing import Callable, TypeVar, Optional, List, Iterable, Tuple, cast

from colorama import Fore

T = TypeVar('T')
R = TypeVar('R')


# Types

def cls_name(o) -> str:
    return o.__class__.__name__


# Optional

def map_opt(f: Callable[[T], R], o: Optional[T]) -> Optional[R]:
    return None if o is None else f(o)


def unwrap(o: Optional[T]) -> T:
    assert o is not None
    return cast(T, o)


def unwrap_or(o: Optional[T], default: T) -> T:
    return default if o is None else o


def filter_none(lst: List[Optional[T]]) -> List[T]:
    return list(filter(lambda e: e is not None, lst))


# Format

def colored_text(txt: str, color: str):
    return color + txt + Fore.RESET


class CodeBuffer:
    """
    Efficient buffer for writing code with indentation.
    """
    indent_str = '    '

    def __init__(self):
        self._buf = StringIO()
        self._new_ln = True
        self.indent_cnt_ = 0

    def write(self, s: str):
        self._try_write_indent()
        self._buf.write(s)

    def writeln(self, s: str = ''):
        self._try_write_indent()
        if len(s) > 0:
            self.write(s)
        self.write('\n')
        self._new_ln = True

    def __str__(self):
        return self._buf.getvalue()

    def indent(self):
        return _Indent(self)

    def write_pos(self, items: List[Callable[[], None]],
                  sep: str = ', ', prefix: str = '(', suffix: str = ')'):
        self.write(prefix)
        for i, callback in enumerate(items):
            if i != 0:
                self.write(sep)
            callback()
        self.write(suffix)

    def write_pos_multi(self, items: List[Callable[[], None]],
                        sep: str = ',', prefix: str = '(', suffix: str = ')'):
        self.writeln(prefix)
        with self.indent():
            for i, callback in enumerate(items):
                callback()
                self.writeln(sep)
        self.write(suffix)

    def write_named(self, items: List[Tuple[str, Callable[[], None]]],
                    sep: str = ', ', prefix: str = '(', suffix: str = ')'):
        self.write(prefix)
        for i, (name, callback) in enumerate(items):
            if i != 0:
                self.write(sep)
            self.write(f'{name}=')
            callback()
        self.write(suffix)

    def write_named_multi(self, items: List[Tuple[str, Callable[[], None]]],
                          sep: str = ',', prefix: str = '(', suffix: str = ')'):
        self.writeln(prefix)
        with self.indent():
            for i, (name, callback) in enumerate(items):
                self.write(f'{name}=')
                callback()
                self.writeln(sep)
        self.write(suffix)

    def _try_write_indent(self):
        if self._new_ln:
            for _ in range(self.indent_cnt_):
                self._buf.write(self.indent_str)
            self._new_ln = False


class _Indent:
    def __init__(self, buf: CodeBuffer):
        self._buf = buf

    def __enter__(self):
        self._buf.indent_cnt_ += 1

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._buf.indent_cnt_ -= 1


class NameGenerator:
    def __init__(self, prefix: str, known: Iterable[str]):
        self._prefix = prefix
        self._known = set(known)
        self._cnt = 0

    def generate(self):
        while True:
            cand = self._prefix + str(self._cnt)
            self._cnt += 1
            if cand not in self._known:
                self._known.add(cand)
                return cand
