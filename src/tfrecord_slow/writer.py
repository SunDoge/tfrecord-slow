import struct
import io
from .utils import make_masked_crc
from typing import Dict, Any
import safetensors


class TfRecordWriter:
    def __init__(self, file: io.FileIO) -> None:
        self._file = file

    @classmethod
    def create(cls, path: str, buffering: int = -1):
        return cls(open(path, mode="wb", buffering=buffering))

    def close(self):
        self._file.close()

    def flush(self):
        self._file.flush()

    def write(self, buf: bytes):
        length = len(buf)
        length_bytes = struct.pack("<Q", length)
        self._file.write(length_bytes)
        self._file.write(make_masked_crc(length_bytes))
        self._file.write(buf)
        self._file.write(make_masked_crc(buf))

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback):
        self.close()
