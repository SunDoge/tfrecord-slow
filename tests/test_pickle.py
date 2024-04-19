from tfrecord_slow.loaders.msgpack_loader import MsgpackTfrecordLoader
from tfrecord_slow.loaders.raw_loader import RawTfrecordLoader
import pickle
import msgspec


def test_raw():
    assert pickle.dumps(RawTfrecordLoader(None))


class Example(msgspec.Struct):
    foo: int


def identity(x):
    return x


def test_msgpack():
    assert pickle.dumps(
        MsgpackTfrecordLoader(
            None,
            Example,
            identity,
        )
    )
