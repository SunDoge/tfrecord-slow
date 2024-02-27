"""
Make a new package for it.
"""

import argparse
import logging
from pathlib import Path
from typing import Optional
from .reader import TfRecordReader

logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    count_parser = subparsers.add_parser("count")
    count_parser.set_defaults(func=count)
    count_parser.add_argument(
        "path", type=Path, help="Path to tfrecord file or directory."
    )
    count_parser.add_argument("-m", "--mask", action="store_const", const="*.tfrec")
    count_parser.add_argument(
        "-c", "--check", action="store_true", help="Check integrity."
    )

    return parser.parse_args()


def count(args):
    path: Path = args.path
    mask: Optional[str] = args.mask
    check_integrity: bool = args.check

    if mask is not None:
        paths = list(path.glob(mask))
    else:
        paths = [path]

    for path in paths:
        with TfRecordReader.open(str(path), check_integrity=check_integrity) as reader:
            num_records = reader.count()
            print(f"file: {path}, records: {num_records}")


def main():
    args = get_args()
    print(args)

    if "func" in args:
        args.func(args)


if __name__ == "__main__":
    main()
