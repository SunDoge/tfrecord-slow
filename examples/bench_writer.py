from tfrecord_slow import TfRecordWriter
from tqdm import tqdm
import numpy as np
from safetensors.numpy import save
import msgspec
from tfrecord_slow.msgpack import enc_hook
from .common import Message, NdArray


def bench(n: int = 1000):
    x = np.random.rand(1024, 1024)
    with TfRecordWriter.create("/tmp/test_writer.tfrec") as writer:
        for _ in tqdm(range(n)):
            writer.write(msgspec.msgpack.encode(Message(NdArray.from_numpy(x))))


bench()
