import numpy as np
import msgspec
from tfrecord_slow.utils.msgpack import NdArray, NdArrayView
from tfrecord_slow.utils.numpy import buffer_to_ndarray


def test_encode_decode():
    x = np.arange(100)
    arr = NdArrayView.from_numpy(x)
    buf = msgspec.msgpack.encode(arr)
    assert len(buf) > 0

    x1 = msgspec.msgpack.decode(buf, type=NdArray).to_numpy()
    assert x.dtype == x1.dtype
    assert np.allclose(x, x1)


def test_buffer_to_ndarray():
    shape = (1, 2, 3, 4)
    dtype = np.float32
    x = np.random.rand(*shape).astype(dtype)
    packed = msgspec.msgpack.encode(x.tobytes())
    unpacked = msgspec.msgpack.decode(packed, type=memoryview)
    x1 = buffer_to_ndarray(unpacked, shape=shape, dtype=dtype)
    assert np.allclose(x, x1)

    x2 = buffer_to_ndarray(unpacked)
    assert x2.dtype == np.uint8
    assert x2.ndim == 1

