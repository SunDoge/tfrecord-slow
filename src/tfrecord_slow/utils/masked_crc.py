import crc32c
import numpy as np
import struct
from typing import Union

MASK = 0xA282EAD8
UINT32_MAX = 0xFFFFFFFF


_Buffer = Union[bytes, bytearray, memoryview]


def make_masked_crc(data: _Buffer) -> bytes:
    crc = np.uint32(crc32c.crc32c(data))
    masked = ((crc >> 15) | (crc << 17)) + MASK
    masked_bytes = struct.pack("<I", masked & UINT32_MAX)
    return masked_bytes


def verify_masked_crc(data: _Buffer, expected: _Buffer) -> bool:
    return make_masked_crc(data) == expected
