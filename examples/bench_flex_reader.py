from tqdm import tqdm
from common import Message
from tfrecord_slow.loaders.raw_loader import RawTfrecordLoader
from flatbuffers import flexbuffers
import numpy as np


def parse(buf):
    # msg = flexbuffers.GetRoot(bytearray(buf))
    msg = flexbuffers.GetRoot(buf)
    # return {"data": msg.X().Data()}
    # x = np.frombuffer(msg.X().DataAsNumpy().data, dtype=msg.X().Dtype().decode())
    # x = x.reshape(msg.X().ShapeAsNumpy())

    # print(msg.IsMap)
    # map = msg.AsMap
    # xmap = map["x"].AsMap

    # x = np.frombuffer(xmap["data"].AsBlob, dtype=xmap["dtype"].AsString)
    # x = x.reshape(xmap["shape"].AsVector.Value)

    # data = msg.AsVector[0].AsBlob
    vec = flexbuffers.Vector(msg._Indirect(), msg._byte_width)
    data = flexbuffers.Blob(vec[0]._Indirect(), vec[0]._byte_width).Bytes

    # print(type(data))
    x = np.frombuffer(data, dtype=np.float64).reshape(1024, 1024)

    return x


def bench(n: int = 1000):
    # with TfRecordReader.open("/tmp/test_writer.tfrec") as reader:
    loader = RawTfrecordLoader([open("/tmp/test_flex_writer.tfrec", "rb")], parse)
    for msg in tqdm(loader, total=n):
        # assert msg.x.to_numpy().shape == (1024, 1024)
        # assert msg.shape == (1024, 1024)
        pass


bench()
