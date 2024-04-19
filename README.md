# tfrecord-slow

TFRecord reader and writer without protobuf.

## Install

### Reader only (without masked crc32 check support)

```shell
pip install tfrecord-slow
```

### Reader with masked crc32 check support

```shell
pip install tfrecord-slow[crc32c]
```

### Writer (must have crc32c installed)

```shell
pip install tfrecord-slow[crc32c]
```

### Use ndarray msgpack support

```shell
pip install tfrecord-slow[msgpack]
```