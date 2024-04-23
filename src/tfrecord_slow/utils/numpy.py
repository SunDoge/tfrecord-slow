import numpy as np
from typing import List


def from_buffer(buf: memoryview, shape: List[int], dtype: np.dtype) -> np.ndarray:
    return np.frombuffer(buf, dtype=dtype).copy().reshape(shape)
