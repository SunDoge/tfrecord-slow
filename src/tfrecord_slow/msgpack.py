import msgspec
from typing import List
import numpy as np

_PREFIX_LENGTHS = {0xC4: 2, 0xC5: 3, 0xC6: 5}


def raw_as_memoryview(raw: msgspec.Raw) -> memoryview:
    raw_view = memoryview(raw)
    prefix_length = _PREFIX_LENGTHS[raw_view[0]]
    return raw_view[prefix_length:]


class NdArray(msgspec.Struct):
    data: msgspec.Raw
    dtype: str
    shape: List[int]

    def to_numpy(self):
        return np.frombuffer(self.as_buffer(), np.dtype(self.dtype)).reshape(self.shape)

    @classmethod
    def from_numpy(cls, arr: np.ndarray):
        return cls(arr.data, arr.dtype.str, arr.shape)

    def as_buffer(self):
        return raw_as_memoryview(self.data)

    @classmethod
    def from_memoryview(cls, buf: memoryview):
        bbuf = buf.cast("B")
        return cls(msgspec.Raw(bbuf), "|u1", [len(bbuf)])


def enc_hook(obj):
    if isinstance(obj, np.ndarray):
        return NdArray.from_numpy(obj)
    else:
        raise NotImplementedError


def dec_hook(ty, obj):
    if ty is np.ndarray:
        print(type(obj))
        import ipdb

        ipdb.set_trace()
        return np.frombuffer(obj["data"], np.dtype(obj["dtype"])).reshape(obj["shape"])
    else:
        raise NotImplementedError
