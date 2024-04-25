"""
Make a new package for it.
"""

import argparse
from pathlib import Path
from typing import Optional
from .reader import TfrecordReader
from loguru import logger
import time


def get_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    add_count_args(subparsers.add_parser("count"))

    return parser.parse_args()


def add_count_args(parser: argparse.ArgumentParser):
    parser.set_defaults(func=count)
    parser.add_argument("path", type=Path, help="Path to tfrecord file or directory.")
    parser.add_argument("-m", "--mask", action="store_const", const="*.tfrec")
    parser.add_argument("-c", "--check", action="store_true", help="Check integrity.")


def count(path: Path, mask: Optional[str], check: bool, **kwargs):
    if mask is not None:
        paths = list(path.glob(mask))
    else:
        paths = [path]

    total = 0

    for path in paths:
        with TfrecordReader.open(str(path), check_integrity=check) as reader:
            start_time = time.perf_counter()
            num_records = reader.count()
            end_time = time.perf_counter()
            records_per_second = num_records / (end_time - start_time)
            logger.info(
                f"file: {path}, records: {num_records}, speed: {records_per_second} rec/s"
            )
            total += num_records

    logger.info("total records: {}", total)


def main():
    args = get_args()
    logger.debug("{}", args)
    if "func" in args:
        args.func(**vars(args))


if __name__ == "__main__":
    main()
