from tfrecord_slow import TfrecordWriter
from tqdm import tqdm
import numpy as np

import flatbuffers
import Data.Message
import Data.NdArray


def create_message(builder: flatbuffers.Builder, x: np.ndarray):
    builder.Clear()

    fbs_dtype = builder.CreateString(x.dtype.str)
    fbs_shape = builder.CreateNumpyVector(np.array(x.shape, dtype=np.int64))
    fbs_data = builder.CreateByteVector(x.data.tobytes())

    Data.NdArray.Start(builder)
    Data.NdArray.AddData(builder, fbs_data)
    Data.NdArray.AddShape(builder, fbs_shape)
    Data.NdArray.AddDtype(builder, fbs_dtype)
    fbs_x = Data.NdArray.End(builder)

    Data.Message.Start(builder)
    Data.Message.AddX(builder, fbs_x)
    fbs_msg = Data.Message.End(builder)

    builder.Finish(fbs_msg)
    return builder.Output()


def bench(n: int = 1000):
    x = np.random.rand(1024, 1024)
    builder = flatbuffers.Builder()
    with TfrecordWriter.create("/tmp/test_fbs_writer.tfrec") as writer:
        for _ in tqdm(range(n)):
            writer.write(create_message(builder, x))


bench()
