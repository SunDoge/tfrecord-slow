import msgspec
from tfrecord_slow.utils.msgpack import NdArrayView, NdArray


class MessageView(msgspec.Struct):
    x: NdArrayView


class Message(msgspec.Struct):
    x: NdArray
