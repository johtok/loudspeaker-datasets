#!/usr/bin/env python3
"""Convert every ExpD MATLAB file into a standardized NumPy dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from convert_pinknoise_to_npz import convert_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert all ExpD MATLAB recordings into .npz archives."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("dtu_34871-NL-Transducers/ExpD"),
        help="Directory containing the ExpD MATLAB files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("numpy_datasets"),
        help="Base directory where numpy datasets are stored.",
    )
    parser.add_argument(
        "--experiment",
        default="ExpD",
        help="Experiment identifier used for the output subdirectory.",
    )
    parser.add_argument(
        "--pattern",
        default="*.mat",
        help="Glob used to select MATLAB files inside --input-dir.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite .npz/.json if they already exist; otherwise skip them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    files = sorted(input_dir.glob(args.pattern))
    if not files:
        raise RuntimeError(f"No files matching {args.pattern!r} found in {input_dir}")

    destination_dir = (args.output_dir / args.experiment).resolve()
    destination_dir.mkdir(parents=True, exist_ok=True)

    converted = 0
    skipped = 0
    for file_path in files:
        dataset_name = file_path.stem
        output_path = destination_dir / f"{dataset_name}.npz"
        metadata_path = output_path.with_suffix(".json")

        if not args.force and (output_path.exists() or metadata_path.exists()):
            print(f"Skipping {dataset_name} (output exists). Use --force to rebuild.")
            skipped += 1
            continue

        convert_file(file_path, output_path, metadata_path)
        print(f"Converted {file_path} -> {output_path}")
        converted += 1

    print(f"Completed conversions. Converted={converted}, skipped={skipped}.")


if __name__ == "__main__":
    main()
