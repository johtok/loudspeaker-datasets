#!/usr/bin/env python3
"""Convert MATLAB recordings into compressed NumPy archives.

This script focuses on the DTU ExpD pink-noise measurement but can be reused
for other MATLAB files by pointing the --input and --dataset-name arguments to
the desired locations.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert MATLAB measurements to NumPy .npz archives."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("dtu_34871-NL-Transducers/ExpD/pinknoise_5Hz-2000Hz_8Vrms.mat"),
        help="Path to the MATLAB .mat file to convert.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("numpy_datasets"),
        help="Directory where .npz files should be stored.",
    )
    parser.add_argument(
        "--dataset-name",
        default="pinknoise_5Hz-2000Hz_8Vrms",
        help="Filename-friendly dataset identifier (without extension).",
    )
    parser.add_argument(
        "--experiment",
        default="ExpD",
        help="Experiment identifier used to segment numpy_datasets.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing .npz and metadata if they are already present.",
    )
    return parser.parse_args()


def flatten_value(
    name: str,
    value: Any,
    flat: dict[str, np.ndarray],
    shapes: dict[str, dict[str, Any]],
) -> None:
    """Recursively flatten MATLAB data into numpy arrays."""
    if isinstance(value, Mapping):
        for sub_key, sub_val in value.items():
            next_name = f"{name}_{sub_key}" if name else str(sub_key)
            flatten_value(next_name, sub_val, flat, shapes)
        return

    if isinstance(value, str):
        arr = np.asarray(value)
        flat[name] = arr
        shapes[name] = {"shape": arr.shape, "dtype": str(arr.dtype)}
        return

    if isinstance(value, Sequence) and not isinstance(
        value, (bytes, bytearray, np.ndarray, str)
    ):
        for idx, item in enumerate(value):
            next_name = f"{name}_{idx}"
            flatten_value(next_name, item, flat, shapes)
        return

    arr = np.asarray(value)
    if arr.dtype == object and arr.size == 1:
        element = arr.item()
        if isinstance(element, Mapping):
            flatten_value(name, element, flat, shapes)
            return
        if isinstance(element, str):
            arr = np.asarray(element)
        elif isinstance(element, Sequence) and not isinstance(
            element, (bytes, bytearray, np.ndarray, str)
        ):
            flatten_value(name, list(element), flat, shapes)
            return
        else:
            arr = np.asarray(element)

    if name in flat:
        raise ValueError(f"duplicate key '{name}' encountered while flattening data.")

    flat[name] = arr
    shapes[name] = {"shape": arr.shape, "dtype": str(arr.dtype)}


def convert_file(
    input_path: Path,
    output_path: Path,
    metadata_path: Path,
) -> dict[str, dict[str, Any]]:
    """Load the .mat file and persist its numpy representation."""
    raw = loadmat(
        input_path,
        squeeze_me=True,
        struct_as_record=False,
        simplify_cells=True,
    )
    flat: dict[str, np.ndarray] = {}
    shapes: dict[str, dict[str, Any]] = {}

    for key, value in raw.items():
        if key.startswith("__"):
            continue
        flatten_value(key, value, flat, shapes)

    if not flat:
        raise RuntimeError(f"No usable arrays were found in {input_path}")

    np.savez_compressed(output_path, **flat)
    metadata_path.write_text(
        json.dumps(
            {
                "source_file": str(input_path),
                "npz_file": str(output_path),
                "converted_at": datetime.now(timezone.utc).isoformat(),
                "arrays": shapes,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return shapes


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    destination_dir = (args.output_dir / args.experiment).resolve()
    destination_dir.mkdir(parents=True, exist_ok=True)
    output_path = destination_dir / f"{args.dataset_name}.npz"
    metadata_path = output_path.with_suffix(".json")

    if not args.force:
        for p in (output_path, metadata_path):
            if p.exists():
                raise FileExistsError(
                    f"{p} already exists. Pass --force to overwrite the existing files."
                )

    shapes = convert_file(input_path, output_path, metadata_path)
    print(f"Wrote {output_path} with {len(shapes)} arrays.")
    print(f"Metadata stored in {metadata_path}")


if __name__ == "__main__":
    main()
