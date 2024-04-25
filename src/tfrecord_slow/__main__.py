"""
Make a new package for it.
"""

import argparse
from pathlib import Path
from typing import Optional
from .reader import TfrecordReader
from .writer import TfrecordWriter
from loguru import logger
import time


def get_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()
    add_count_args(subparsers.add_parser("count", help="Count the number of records."))
    add_split_args(
        subparsers.add_parser("split", help="Split one tfrecord file into parts.")
    )
    add_merge_args(
        subparsers.add_parser("merge", help="Merge tfrecord files into one.")
    )

    return parser.parse_args()


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


def add_count_args(parser: argparse.ArgumentParser):
    parser.set_defaults(func=count)
    parser.add_argument("path", type=Path, help="Path to tfrecord file or directory.")
    parser.add_argument("-m", "--mask", action="store_const", const="*.tfrec")
    parser.add_argument("-c", "--check", action="store_true", help="Check integrity.")


def split(
    path: Path,
    output: Optional[Path],
    num_samples: int,
    width: int,
    check: bool,
    **kwargs,
):
    if output is None:
        output = path.parent

    suffix_template = "{:0{width}d}"
    stem = path.stem
    extension = path.suffix

    count = 0
    writer: TfrecordWriter = None
    with TfrecordReader.open(str(path), check_integrity=check) as reader:
        for buf in reader:
            if count % num_samples == 0:
                if writer is not None:
                    writer.close()
                    logger.info("close writer")

                index = count // num_samples
                suffix = suffix_template.format(index, width=width)
                filename = output / f"{stem}-{suffix}{extension}"
                writer = TfrecordWriter.create(str(filename))
                logger.info("open writer: {}", filename)

            writer.write(buf)
            count += 1

    writer.close()


def add_split_args(parser: argparse.ArgumentParser):
    parser.set_defaults(func=split)
    parser.add_argument("path", type=Path, help="Path to tfrecord file.")
    parser.add_argument("-o", "--output", type=Path, help="Output dir for file parts.")
    parser.add_argument(
        "-n", "--num-samples", type=int, default=1024, help="num samples per file."
    )
    parser.add_argument(
        "-w", "--width", type=int, default=4, help="Num numbers in file name."
    )
    parser.add_argument("-c", "--check", action="store_true", help="Check integrity.")


def merge():
    pass


def add_merge_args(parser: argparse.ArgumentParser):
    pass


def main():
    args = get_args()
    logger.debug("{}", args)
    if "func" in args:
        args.func(**vars(args))


if __name__ == "__main__":
    main()
