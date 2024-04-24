from tqdm import tqdm
from common import Message
from tfrecord_slow.loaders.raw_loader import RawTfrecordLoader
import Data.Message
import numpy as np


def parse(buf):
    msg = Data.Message.Message.GetRootAs(buf)
    # return {"data": msg.X().Data()}
    x = np.frombuffer(msg.X().DataAsNumpy().data, dtype=msg.X().Dtype().decode())
    x = x.reshape(msg.X().ShapeAsNumpy())
    return x


def bench(n: int = 1000):
    # with TfRecordReader.open("/tmp/test_writer.tfrec") as reader:
    loader = RawTfrecordLoader([open("/tmp/test_fbs_writer.tfrec", "rb")], parse)
    for msg in tqdm(loader, total=n):
        # assert msg.x.to_numpy().shape == (1024, 1024)
        assert msg.shape == (1024, 1024)


bench()
