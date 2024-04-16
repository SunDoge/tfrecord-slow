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

    def to_view(self):
        return NdArrayView(memoryview(self.data), self.dtype, self.shape)


class NdArrayView(msgspec.Struct):
    data: memoryview
    dtype: str
    shape: List[int]

    @classmethod
    def from_numpy(cls, arr: np.ndarray):
        """
        Args:
            arr: numpy ndarray
            copy: copy the memory or keep it as a memoryview
        """
        return cls(arr.data, arr.dtype.str, arr.shape)

    def to_owned(self):
        return NdArray(bytearray(self.data), self.dtype, self.shape)
