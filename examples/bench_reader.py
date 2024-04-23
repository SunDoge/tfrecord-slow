from tqdm import tqdm
from common import Message
from tfrecord_slow.loaders.msgpack_loader import MsgpackTfrecordLoader


def bench(n: int = 1000):
    # with TfRecordReader.open("/tmp/test_writer.tfrec") as reader:
    loader = MsgpackTfrecordLoader(
        [open("/tmp/test_writer.tfrec", "rb")], Message, check_integrity=True
    )
    for msg in tqdm(loader, total=n):
        assert msg.x.to_numpy().shape == (1024, 1024)


bench()
