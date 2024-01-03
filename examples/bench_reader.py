from tfrecord_slow import TfRecordReader
from tqdm import tqdm
import numpy as np
import msgspec
from tfrecord_slow.msgpack import NdArray


class Message(msgspec.Struct):
    x: NdArray


def bench(n: int = 1000):
    with TfRecordReader.open("/tmp/test_writer.tfrec") as reader:
        for data in tqdm(reader, total=n):
            msg = msgspec.msgpack.decode(data, type=Message)
            assert msg.x.to_numpy().shape == (1024, 1024)


bench()
