import sys
from io import StringIO
from multiprocessing import Manager, Process, Queue, Semaphore
from queue import Empty
from threading import Thread
from time import sleep
from typing import Callable, TypeVar, Optional, List, Iterable, Tuple, Generic, Any, Dict, \
    NamedTuple, cast, Set

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


# Dict

def inc_cnt(cnt_map: Dict[T, int], key: T):
    if key in cnt_map:
        cnt_map[key] += 1
    else:
        cnt_map[key] = 1


# Reference

class Ref(Generic[T]):
    """
    Wrapper for object that support equality and hashing w.r.t. its reference (id), no matter
    whether it overrides `__eq__` and `__hash__` or not.
    """

    def __init__(self, obj: T):
        self.obj_ = obj

    def get(self):
        return self.obj_

    def __eq__(self, other: 'Ref[T]'):
        return self.obj_ is other.obj_

    def __hash__(self):
        return hash(id(self.obj_))

    def __repr__(self):
        return hex(id(self.obj_))


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
        class Indent:
            def __init__(self, buf: CodeBuffer):
                self._buf = buf

            def __enter__(self):
                self._buf.indent_cnt_ += 1

            def __exit__(self, exc_type, exc_val, exc_tb):
                self._buf.indent_cnt_ -= 1

        return Indent(self)

    def write_pos(self, items: Iterable[Callable[[], None]],
                  sep: str = ', ', prefix: str = '(', suffix: str = ')'):
        self.write(prefix)
        for i, callback in enumerate(items):
            if i != 0:
                self.write(sep)
            callback()
        self.write(suffix)

    def write_pos_multi(self, items: Iterable[Callable[[], None]],
                        sep: str = ',', prefix: str = '(', suffix: str = ')'):
        self.writeln(prefix)
        with self.indent():
            for i, callback in enumerate(items):
                callback()
                self.writeln(sep)
        self.write(suffix)

    def write_named(self, items: Iterable[Tuple[str, Callable[[], None]]],
                    sep: str = ', ', prefix: str = '(', suffix: str = ')'):
        self.write(prefix)
        for i, (name, callback) in enumerate(items):
            if i != 0:
                self.write(sep)
            self.write(f'{name}=')
            callback()
        self.write(suffix)

    def write_named_multi(self, items: Iterable[Tuple[str, Callable[[], None]]],
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


class NameGenerator:
    def __init__(self, prefix: str, known: Optional[Iterable[str]] = None):
        self._prefix = prefix
        self._known: Set[str] = set()
        if known is not None:
            self._check = True
            self._known = set(known)
        else:
            self._check = False
        self._cnt = 0

    def generate(self):
        while True:
            cand = self._prefix + str(self._cnt)
            self._cnt += 1
            if not self._check:
                return cand
            if cand not in self._known:
                self._known.add(cand)
                return cand


# Multiprocessing

class ProcessResult(NamedTuple):
    exitcode: int
    ret: Dict[str, Any]
    stdout: str
    stderr: str


def run_process(f: Callable[[Any], Dict[str, Any]], args: Tuple[Any, ...]):
    # Define concurrent queue reader
    class QueueReader:
        def __init__(self, queue: Queue):
            self._queue = queue
            self._buf = StringIO()
            self._sema = Semaphore(value=1)
            self._sema.acquire(block=True)

            def try_get():
                while True:
                    try:
                        s = self._queue.get(block=False)
                        self._buf.write(s)
                    except Empty:
                        if self._sema.acquire(block=False):
                            break
                        sleep(0.01)

            self._thread = Thread(target=try_get, args=())
            self._thread.start()

        def join(self):
            self._sema.release()
            self._thread.join()
            return self._buf.getvalue()

    # Prepare data
    ret: Dict[str, Any] = Manager().dict()
    out_q, err_q = Queue(), Queue()
    out_read, err_read = QueueReader(out_q), QueueReader(err_q)

    # Start process
    ps = Process(target=_ps_work, args=(f, args, ret, out_q, err_q))
    ps.start()
    ps.join(timeout=60)

    return ProcessResult(ps.exitcode, dict(ret), out_read.join(), err_read.join())


def _ps_work(f: Callable[[Any], Dict[str, Any]], args: Tuple[Any, ...], ret: Dict[str, Any],
             out: Queue, err: Queue):
    class QueueIO:
        def __init__(self, queue: Queue):
            self._q = queue

        def write(self, s: str):
            self._q.put(s)

    # Redirect stdout and stderr to queues
    sys.stdout = QueueIO(out)
    sys.stderr = QueueIO(err)

    # Call function and write to dict proxy
    ret.update(f(*args))

    # Restore stdout and stderr
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
