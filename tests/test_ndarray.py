import numpy as np
import msgspec
from tfrecord_slow.msgpack import NdArray


def test_encode_decode():
    x = np.arange(100)
    arr = NdArray.from_numpy(x)
    buf = msgspec.msgpack.encode(arr)
    assert len(buf) > 0

    x1 = msgspec.msgpack.decode(buf, type=NdArray).to_numpy()
    assert x.dtype == x1.dtype
    assert np.allclose(x, x1)
