from tfrecord_slow.utils.masked_crc import make_masked_crc, verify_masked_crc
import struct


IN_BUF = b"1234567890asdfghjkl"
OUT_INT = 4134599430

WRONG_BUF = b"ddddssssaaaaa"


def test_make():
    out_buf = make_masked_crc(IN_BUF)
    assert struct.unpack("<I", out_buf)[0] == OUT_INT


def test_verify():
    assert verify_masked_crc(IN_BUF, struct.pack("<I", OUT_INT))
    assert not verify_masked_crc(WRONG_BUF, struct.pack("<I", OUT_INT))
