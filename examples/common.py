import msgspec
from tfrecord_slow.msgpack import NdArray


class Message(msgspec.Struct):
    x: NdArray
