import numpy as np
from typing import List, Optional


def from_buffer(
    buf: memoryview,
    shape: List[int] | None = None,
    dtype: np.dtype = np.uint8,
    copy: bool = True,
) -> np.ndarray:
    x = np.frombuffer(buf, dtype=dtype)
    if copy:
        x = x.copy()
    if shape is not None:
        x = x.reshape(shape)
    return x
