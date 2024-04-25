import io
from typing import Optional
import struct
import logging


logger = logging.getLogger(__name__)


def _always_return_true(*args, **kwargs):
    return True


class TfrecordReader:
    def __init__(self, file: io.BufferedIOBase, check_integrity: bool = False) -> None:
        self._file = file

        if check_integrity:
            from tfrecord_slow.utils.masked_crc import verify_masked_crc

            self._verify_masked_crc32 = verify_masked_crc
        else:
            self._verify_masked_crc32 = _always_return_true

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

        if not self._verify_masked_crc32(self._length_bytes, self._crc_bytes):
            raise RuntimeError("Crc32 check failed.")

        (length,) = struct.unpack("<Q", self._length_bytes)
        if length > len(self._data_bytes):
            self._data_bytes = self._data_bytes.zfill(length * 2)

        data_bytes_view = memoryview(self._data_bytes)[:length]

        if self._file.readinto(data_bytes_view) != length:
            raise RuntimeError("Invalid tfrecord file: failed to read the record.")
        if self._file.readinto(self._crc_bytes) != 4:
            raise RuntimeError("Invalid tfrecord file: failed to read the end token.")

        if not self._verify_masked_crc32(data_bytes_view, self._crc_bytes):
            raise RuntimeError("Crc32 check failed.")

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

    def count(self) -> int:
        n = 0
        for _ in self:
            n += 1
        return n
