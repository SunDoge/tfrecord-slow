import numpy as np
from typing import Optional, Sequence


def buffer_to_ndarray(
    buf: memoryview,
    shape: Optional[Sequence[int]] = None,
    dtype: np.dtype = np.uint8,
    copy: bool = True,
) -> np.ndarray:
    x = np.frombuffer(buf, dtype=dtype)
    if copy:
        x = x.copy()
    if shape is not None:
        x = x.reshape(shape)
    return x
