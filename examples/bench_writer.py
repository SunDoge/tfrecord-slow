from tfrecord_slow import TfrecordWriter
from tqdm import tqdm
import numpy as np

import msgspec
from common import MessageView
from tfrecord_slow.utils.msgpack import NdArrayView


def bench(n: int = 1000):
    x = np.random.rand(1024, 1024)
    with TfrecordWriter.create("/tmp/test_writer.tfrec") as writer:
        for _ in tqdm(range(n)):
            writer.write(msgspec.msgpack.encode(MessageView(NdArrayView.from_numpy(x))))


bench()
