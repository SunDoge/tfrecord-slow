# automatically generated by the FlatBuffers compiler, do not modify

# namespace: Data

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class Message(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = Message()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsMessage(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    # Message
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # Message
    def X(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            x = self._tab.Indirect(o + self._tab.Pos)
            from Data.NdArray import NdArray
            obj = NdArray()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

def Start(builder): builder.StartObject(1)
def MessageStart(builder):
    """This method is deprecated. Please switch to Start."""
    return Start(builder)
def AddX(builder, x): builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(x), 0)
def MessageAddX(builder, x):
    """This method is deprecated. Please switch to AddX."""
    return AddX(builder, x)
def End(builder): return builder.EndObject()
def MessageEnd(builder):
    """This method is deprecated. Please switch to End."""
    return End(builder)