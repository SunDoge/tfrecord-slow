from typing import TypeVar, Iterable, Iterator, Type, Tuple
import msgspec
from io import BufferedIOBase
from tfrecord_slow.reader import TfRecordReader

T = TypeVar("T")


class TfRecordLoader:
    def __init__(
        self,
        datapipe: Iterable[BufferedIOBase],
        spec: Type[T],
        check_integrity: bool = False,
    ):
        self.datapipe = datapipe
        self.spec = spec
        self.check_integrity = check_integrity

    def __iter__(self) -> Iterator[T]:
        for fp in self.datapipe:
            reader = TfRecordReader(fp, check_integrity=self.check_integrity)
            for buf in reader:
                example = msgspec.msgpack.decode(buf, type=self.spec)
                yield example
