import msgspec
from typing import List
import numpy as np


class NdArray(msgspec.Struct):
    data: bytearray  # Make it mutable.
    dtype: str
    shape: List[int]

    def to_numpy(self):
        """
        Attention!!! the internal buffer is mutable
        """
        return np.frombuffer(self.data, np.dtype(self.dtype)).reshape(self.shape)

    @classmethod
    def from_numpy(cls, arr: np.ndarray):
        return cls(arr.data, arr.dtype.str, arr.shape)

    @classmethod
    def from_memoryview(cls, buf: memoryview):
        bbuf = buf.cast("B")
        return cls(bbuf, "|u1", [len(bbuf)])


def enc_hook(obj):
    if isinstance(obj, np.ndarray):
        return NdArray.from_numpy(obj)
    else:
        raise NotImplementedError


def dec_hook(ty, obj):
    if ty is np.ndarray:
        return np.frombuffer(obj["data"], np.dtype(obj["dtype"])).reshape(obj["shape"])
    else:
        raise NotImplementedError
