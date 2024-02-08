import crc32c
import numpy as np
import struct

MASK = 0xA282EAD8
UINT32_MAX = 0xFFFFFFFF


def make_masked_crc(data: bytes) -> bytes:
    crc = np.uint32(crc32c.crc32c(data))
    masked = ((crc >> 15) | (crc << 17)) + MASK
    masked_bytes = struct.pack("<I", masked & UINT32_MAX)
    return masked_bytes


