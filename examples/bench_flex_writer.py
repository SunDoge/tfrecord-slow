from tfrecord_slow import TfrecordWriter
from tqdm import tqdm
import numpy as np

from flatbuffers import flexbuffers
import Data.Message
import Data.NdArray


def create_message(builder: flexbuffers.Builder, x: np.ndarray):
    builder.Clear()

    # with builder.Map() as map:
    #     map.Key("x")
    #     with map.Map() as map:
    #         map.Key("data")
    #         map.Blob(x.tobytes())

    #         map.Key("shape")
    #         with map.Vector() as vec:
    #             for s in x.shape:
    #                 vec.Int(s)

    #         map.Key("dtype")
    #         map.String(x.dtype.str)
    with builder.Vector() as vec:
        vec.Blob(x.tobytes())
        vec.String(x.dtype.str)



    return builder.Finish()

    # fbs_dtype = builder.CreateString(x.dtype.str)
    # fbs_shape = builder.CreateNumpyVector(np.array(x.shape, dtype=np.int64))
    # fbs_data = builder.CreateByteVector(x.data.tobytes())

    # Data.NdArray.Start(builder)
    # Data.NdArray.AddData(builder, fbs_data)
    # Data.NdArray.AddShape(builder, fbs_shape)
    # Data.NdArray.AddDtype(builder, fbs_dtype)
    # fbs_x = Data.NdArray.End(builder)

    # Data.Message.Start(builder)
    # Data.Message.AddX(builder, fbs_x)
    # fbs_msg = Data.Message.End(builder)

    # builder.Finish(fbs_msg)
    # return builder.Output()


def bench(n: int = 1000):
    x = np.random.rand(1024, 1024)
    # builder = flatbuffers.Builder()
    builder = flexbuffers.Builder()
    with TfrecordWriter.create("/tmp/test_flex_writer.tfrec") as writer:
        for _ in tqdm(range(n)):
            writer.write(create_message(builder, x))


bench()
