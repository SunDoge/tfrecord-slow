from typing import TypeVar, Iterable, Iterator, Callable, Generic
from io import BufferedIOBase
from tfrecord_slow.reader import TfrecordReader

T = TypeVar("T")


def _default_func(buf: memoryview):
    return buf


class RawTfrecordLoader(Generic[T]):
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
            reader = TfrecordReader(fp, check_integrity=self.check_integrity)
            for buf in reader:
                example = self.func(buf)
                yield example
