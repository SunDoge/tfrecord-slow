import io
from typing import Optional
import struct
import logging

logger = logging.getLogger(__name__)


class TfRecordReader:
    def __init__(self, file: io.BufferedIOBase, check_integrity: bool = False) -> None:
        self._file = file
        self._check_integrity = check_integrity

        self._length_bytes = bytearray(8)
        self._crc_bytes = bytearray(4)
        self._data_bytes = bytearray(1024 * 1024)

    @classmethod
    def open(cls, path: str, check_integrity: bool = False):
        return cls(open(path, "rb"), check_integrity=check_integrity)

    def close(self):
        logger.debug("Close file: %s", self._file)
        self._file.close()

    def read(self) -> Optional[memoryview]:
        bytes_read = self._file.readinto(self._length_bytes)
        if bytes_read == 0:
            return
        elif bytes_read != 8:
            raise RuntimeError("Invalid tfrecord file: failed to read the record size.")

        if self._file.readinto(self._crc_bytes) != 4:
            raise RuntimeError("Invalid tfrecord file: failed to read the start token.")

        (length,) = struct.unpack("<Q", self._length_bytes)
        if length > len(self._data_bytes):
            self._data_bytes = self._data_bytes.zfill(length * 2)

        data_bytes_view = memoryview(self._data_bytes)[:length]
        if self._file.readinto(data_bytes_view) != length:
            raise RuntimeError("Invalid tfrecord file: failed to read the record.")
        if self._file.readinto(self._crc_bytes) != 4:
            raise RuntimeError("Invalid tfrecord file: failed to read the end token.")
        return data_bytes_view

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback):
        self.close()

    def __iter__(self):
        while True:
            data = self.read()
            if data is not None:
                yield data
            else:
                break

    def count(self):
        return sum(1 for _ in self)
