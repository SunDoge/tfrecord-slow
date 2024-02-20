from typing import TypeVar, Iterable, Iterator, Callable
from io import BufferedIOBase
from tfrecord_slow.reader import TfRecordReader

T = TypeVar("T")


def _default_func(buf: memoryview):
    return buf.tobytes()


class TfRecordLoader:
    def __init__(
        self,
        datapipe: Iterable[BufferedIOBase],
        func: Callable[[memoryview], T] = _default_func,
        check_integrity: bool = False,
    ):
        self.datapipe = datapipe
        self.check_integrity = check_integrity
        self.func = func

    def __iter__(self) -> Iterator[T]:
        for fp in self.datapipe:
            reader = TfRecordReader(fp, check_integrity=self.check_integrity)
            for buf in reader:
                example = self.func(buf)
                yield example
