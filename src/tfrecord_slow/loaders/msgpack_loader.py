from typing import TypeVar, Iterable, Iterator, Generic, Type
from io import BufferedIOBase
from tfrecord_slow.reader import TfrecordReader
import msgspec


T = TypeVar("T", bound=msgspec.Struct)


class MsgpackTfrecordLoader(Generic[T]):
    def __init__(
        self,
        datapipe: Iterable[BufferedIOBase],
        spec: Type[T],
        check_integrity: bool = False,
    ):
        """
        Args:
            datapipe: iter of files
            spec: msgspec.Struct, must have owned memory
        """
        self.datapipe = datapipe
        self.check_integrity = check_integrity
        self.spec = spec

    def __iter__(self) -> Iterator[T]:
        decoder = msgspec.msgpack.Decoder(type=self.spec)
        for fp in self.datapipe:
            reader = TfrecordReader(fp, check_integrity=self.check_integrity)
            for buf in reader:
                example = decoder.decode(buf)
                yield example
