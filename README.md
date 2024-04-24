# tfrecord-slow

[![Github Actions](https://img.shields.io/github/actions/workflow/status/SunDoge/tfrecord-slow/python-package.yml?branch=main&style=for-the-badge)](https://github.com/SunDoge/tfrecord-slow/actions/workflows/python-package.yml)
[![Pypi](https://img.shields.io/pypi/v/tfrecord-slow?style=for-the-badge)](https://pypi.org/project/tfrecord-slow/)

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