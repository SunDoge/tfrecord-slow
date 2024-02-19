import argparse
import logging
from pathlib import Path
from typing import Optional
from .reader import TfRecordReader

logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    check_parser = subparsers.add_parser("check")
    check_parser.set_defaults(func=check)
    check_parser.add_argument(
        "path", type=Path, help="Path to tfrecord file or directory."
    )
    check_parser.add_argument("-m", "--mask", action="store_const", const="*.tfrec")

    return parser.parse_args()


def check(args):
    path: Path = args.path
    mask: Optional[str] = args.mask

    if mask is not None:
        paths = list(path.glob(mask))
    else:
        paths = [path]

    for path in paths:
        with TfRecordReader.open(str(path), check_integrity=True) as reader:
            num_records = reader.count()
            logger.info(f"file: {path}, records: {num_records}")


def main():
    args = get_args()
    print(args)

    if "func" in args:
        args.func(args)


if __name__ == "__main__":
    main()
