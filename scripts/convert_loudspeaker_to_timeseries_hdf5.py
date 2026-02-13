#!/usr/bin/env python3
"""Convert loudspeaker datasets into a timeseries-style HDF5 file.

The output HDF5 schema mirrors the format used by Loudspeaker.py:

- root dataset: `time` (shape: [n_time])
- group: `<group>` (default: `nonlinear_loudspeaker`)
- subgroups: `sample_000001`, `sample_000002`, ...
    - datasets: `input_signal` (shape: [n_time]), `states` (shape: [n_time, 3])

State order is `[current_a, displacement_m, velocity_m_s]` and `input_signal` is
the terminal voltage in volts.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import asdf
import h5py
import numpy as np
from scipy.io import loadmat


STATE_NAMES = ("current_a", "displacement_m", "velocity_m_s")
INPUT_NAME = "voltage_v"


@dataclass(frozen=True)
class Sample:
    input_signal: np.ndarray  # (n_time,)
    states: np.ndarray  # (n_time, 3) -> [i, x, v]
    attrs: dict[str, Any]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _as_h5_str(value: str) -> np.bytes_:
    return np.bytes_(value.encode("utf-8"))


def _time_vector(sample_rate_hz: int, duration_s: float) -> np.ndarray:
    n_time = int(round(duration_s * sample_rate_hz)) + 1
    return np.linspace(0.0, duration_s, n_time, dtype=np.float64)


def _pad_or_trim_1d(x: np.ndarray, n: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.shape[0] == n:
        return x
    if x.shape[0] > n:
        return x[:n]
    pad = n - x.shape[0]
    return np.pad(x, (0, pad), mode="edge")


def _windows_1d(x: np.ndarray, n: int, stride: int, include_tail: bool) -> Iterable[tuple[int, np.ndarray]]:
    x = np.asarray(x).reshape(-1)
    if stride <= 0:
        raise ValueError("stride must be >= 1")
    x_len = int(x.shape[0])
    if x_len <= n:
        yield 0, _pad_or_trim_1d(x, n)
        return

    last_full_start: int | None = None
    for start in range(0, x_len - n + 1, stride):
        last_full_start = start
        yield start, x[start : start + n]

    if not include_tail:
        return

    tail_start = ((x_len - 1) // stride) * stride
    if tail_start != last_full_start:
        yield tail_start, _pad_or_trim_1d(x[tail_start:], n)


def _write_h5(
    output_path: Path,
    *,
    time: np.ndarray,
    group_name: str,
    samples: list[Sample],
    root_attrs: dict[str, Any],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as f:
        f.create_dataset("time", data=np.asarray(time, dtype=np.float64))

        root = {
            "created": _as_h5_str(_utc_now_iso()),
            "description": _as_h5_str("Measured loudspeaker timeseries dataset"),
            "sample_rate_hz": int(round(1.0 / float(time[1] - time[0]))),
            "duration_per_sample_s": float(time[-1] - time[0]),
            "float_precision": _as_h5_str("Float64"),
            "input_name": _as_h5_str(INPUT_NAME),
            "state_names_json": _as_h5_str(json.dumps(list(STATE_NAMES))),
        }
        root.update(root_attrs)
        for k, v in root.items():
            f.attrs[k] = v

        group = f.create_group(group_name)
        for idx, sample in enumerate(samples, start=1):
            sg = group.create_group(f"sample_{idx:06d}")
            sg.create_dataset("input_signal", data=sample.input_signal.astype(np.float64, copy=False))
            sg.create_dataset("states", data=sample.states.astype(np.float64, copy=False))
            sg.attrs["sample_idx"] = int(sample.attrs.get("sample_idx", idx))
            sg.attrs["seed"] = int(sample.attrs.get("seed", 0))
            for k, v in sample.attrs.items():
                if k in {"sample_idx", "seed"}:
                    continue
                if isinstance(v, str):
                    sg.attrs[k] = _as_h5_str(v)
                elif isinstance(v, (int, np.integer)):
                    sg.attrs[k] = int(v)
                elif isinstance(v, (float, np.floating)):
                    sg.attrs[k] = float(v)
                elif v is None:
                    sg.attrs[k] = _as_h5_str("null")
                else:
                    sg.attrs[k] = _as_h5_str(json.dumps(v, default=str))


def _expd_samples(
    mat_path: Path,
    *,
    duration_s: float,
    stride_s: float,
    include_tail: bool,
) -> tuple[int, list[Sample], dict[str, Any]]:
    raw = loadmat(mat_path, squeeze_me=True, struct_as_record=False, simplify_cells=True)
    required = ("voltage", "current", "displacement", "velocity", "sample_rate")
    missing = [k for k in required if k not in raw]
    if missing:
        raise KeyError(f"{mat_path} missing required keys: {missing}")

    sample_rate_hz = int(raw["sample_rate"])
    n_time = int(round(duration_s * sample_rate_hz)) + 1
    stride = int(round(stride_s * sample_rate_hz))

    voltage = np.asarray(raw["voltage"], dtype=np.float64).reshape(-1)
    current = np.asarray(raw["current"], dtype=np.float64).reshape(-1)
    displacement_m = np.asarray(raw["displacement"], dtype=np.float64).reshape(-1)
    velocity_m_s = np.asarray(raw["velocity"], dtype=np.float64).reshape(-1)

    samples: list[Sample] = []
    for start, u in _windows_1d(voltage, n_time, stride, include_tail):
        x = _pad_or_trim_1d(displacement_m[start:], n_time)
        v = _pad_or_trim_1d(velocity_m_s[start:], n_time)
        i = _pad_or_trim_1d(current[start:], n_time)
        states = np.stack([i, x, v], axis=1).astype(np.float64, copy=False)
        samples.append(
            Sample(
                input_signal=_pad_or_trim_1d(u, n_time),
                states=states,
                attrs={
                    "sample_idx": len(samples) + 1,
                    "seed": 0,
                    "source_file": str(mat_path),
                    "start_index": int(start),
                    "experiment": "ExpD",
                    "signal": "pinknoise",
                },
            )
        )

    root_attrs = {
        "source_files_json": _as_h5_str(json.dumps([str(mat_path)])),
        "source_kind": _as_h5_str("matlab"),
    }
    return sample_rate_hz, samples, root_attrs


def _asdf_measurement_to_signals(measurement: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    names = list(measurement["input_names"])
    units = list(measurement["input_units"])
    input_data = np.asarray(measurement["input_data"], dtype=np.float64)
    if input_data.ndim != 2 or input_data.shape[1] != len(names):
        raise ValueError("input_data must be 2D with columns matching input_names")

    def col(name: str) -> tuple[np.ndarray, str]:
        idx = names.index(name)
        return input_data[:, idx], units[idx]

    voltage, _ = col("voltage")
    current, _ = col("current")
    displacement, disp_unit = col("displacement")
    velocity, _ = col("velocity")

    if disp_unit.lower() == "mm":
        displacement_m = displacement * 1e-3
    else:
        displacement_m = displacement

    return voltage, current, displacement_m, velocity


def _asdf_measurements(asdf_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    with asdf.open(asdf_path, lazy_load=False) as af:
        measurements = list(af.tree.get("measurements", []))

    if not measurements:
        raise RuntimeError(f"No measurements found in {asdf_path}")

    root_attrs = {
        "source_files_json": _as_h5_str(json.dumps([str(asdf_path)])),
        "source_kind": _as_h5_str("asdf"),
    }
    return measurements, root_attrs


def _measurement_duration_s(measurement: dict[str, Any]) -> float:
    samplerate = int(measurement["samplerate"])
    input_data = np.asarray(measurement["input_data"], dtype=np.float64)
    n_time = int(input_data.shape[0])
    return (n_time - 1) / samplerate


def _variant_label(signal: str, duration_s: int) -> str:
    return f"{signal}_{duration_s:d}s"


def _vrms_label(vrms_v: float | None) -> tuple[str, int | None]:
    if vrms_v is None:
        return "vrms_unknown", None
    mv = int(round(float(vrms_v) * 1e3))
    return f"vrms_{mv:d}mV", mv


def _write_partitioned_asdf_h5(
    output_path: Path,
    *,
    group_name: str,
    measurements: list[dict[str, Any]],
    root_attrs: dict[str, Any],
    source_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample_rate_values = {int(m["samplerate"]) for m in measurements}
    if len(sample_rate_values) != 1:
        raise ValueError(f"Measurements contain multiple samplerates: {sorted(sample_rate_values)}")
    sample_rate_hz = next(iter(sample_rate_values))

    with h5py.File(output_path, "w") as f:
        root = {
            "created": _as_h5_str(_utc_now_iso()),
            "description": _as_h5_str("Measured loudspeaker timeseries dataset"),
            "float_precision": _as_h5_str("Float64"),
            "input_name": _as_h5_str(INPUT_NAME),
            "state_names_json": _as_h5_str(json.dumps(list(STATE_NAMES))),
            "layout": _as_h5_str("partitioned_variants"),
        }
        root.update(root_attrs)
        for k, v in root.items():
            f.attrs[k] = v

        group = f.create_group(group_name)

        # Group by (signal, rounded duration, samplerate)
        variants: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
        for m in measurements:
            signal = str(m.get("signal", "unknown"))
            duration_s = int(round(_measurement_duration_s(m)))
            key = (signal, duration_s, int(m["samplerate"]))
            variants.setdefault(key, []).append(m)

        for (signal, duration_s, samplerate), items in sorted(variants.items(), key=lambda k: str(k[0])):
            variant_name = _variant_label(signal, duration_s)
            variant_group = group.create_group(variant_name)
            time = _time_vector(samplerate, duration_s)
            variant_group.create_dataset("time", data=np.asarray(time, dtype=np.float64))
            variant_group.attrs["signal"] = _as_h5_str(signal)
            variant_group.attrs["duration_s"] = float(duration_s)
            variant_group.attrs["sample_rate_hz"] = int(samplerate)

            seeds = sorted({int(m.get("seed", 0)) for m in items})
            seed_to_realization = {seed: f"realization_{idx + 1:d}" for idx, seed in enumerate(seeds)}

            vrms_counters: dict[tuple[str, str], int] = {}

            for m_idx, m in enumerate(items):
                seed = int(m.get("seed", 0))
                realization_name = seed_to_realization[seed]
                realization_group = variant_group.require_group(realization_name)
                realization_group.attrs["seed"] = int(seed)

                vrms_v = m.get("vrms")
                vrms_group_name, vrms_mv = _vrms_label(vrms_v)
                vrms_group = realization_group.require_group(vrms_group_name)
                if vrms_v is not None:
                    vrms_group.attrs["vrms_v"] = float(vrms_v)
                if vrms_mv is not None:
                    vrms_group.attrs["vrms_mv"] = int(vrms_mv)

                voltage, current, displacement_m, velocity_m_s = _asdf_measurement_to_signals(m)
                n_time = time.shape[0]
                u = _pad_or_trim_1d(voltage, n_time)
                i = _pad_or_trim_1d(current, n_time)
                x = _pad_or_trim_1d(displacement_m, n_time)
                v = _pad_or_trim_1d(velocity_m_s, n_time)
                states = np.stack([i, x, v], axis=1).astype(np.float64, copy=False)

                counter_key = (variant_name, realization_name, vrms_group_name)
                vrms_counters[counter_key] = vrms_counters.get(counter_key, 0) + 1
                sample_idx = vrms_counters[counter_key]

                sample_group = vrms_group.create_group(f"sample_{sample_idx:06d}")
                sample_group.create_dataset("input_signal", data=u.astype(np.float64, copy=False))
                sample_group.create_dataset("states", data=states.astype(np.float64, copy=False))

                attrs: dict[str, Any] = {
                    "sample_idx": sample_idx,
                    "seed": seed,
                    "source_file": str(source_path),
                    "measurement_index": int(m_idx),
                    "segment_index": 0,
                    "start_index": 0,
                    "original_duration_s": float(_measurement_duration_s(m)),
                }
                for k, v in m.items():
                    if k in {"input_data", "input_data_var", "output_data"}:
                        continue
                    attrs[k] = v
                for k, v in attrs.items():
                    if k in {"sample_idx", "seed"}:
                        continue
                    if isinstance(v, str):
                        sample_group.attrs[k] = _as_h5_str(v)
                    elif isinstance(v, (int, np.integer)):
                        sample_group.attrs[k] = int(v)
                    elif isinstance(v, (float, np.floating)):
                        sample_group.attrs[k] = float(v)
                    elif v is None:
                        sample_group.attrs[k] = _as_h5_str("null")
                    else:
                        sample_group.attrs[k] = _as_h5_str(json.dumps(v, default=str))
                sample_group.attrs["sample_idx"] = int(sample_idx)
                sample_group.attrs["seed"] = int(seed)


def _asdf_samples(
    asdf_path: Path,
    *,
    duration_s: float,
    stride_s: float,
    include_tail: bool,
) -> tuple[int, list[Sample], dict[str, Any]]:
    measurements, root_attrs = _asdf_measurements(asdf_path)

    sample_rate_values = {int(m["samplerate"]) for m in measurements}
    if len(sample_rate_values) != 1:
        raise ValueError(f"{asdf_path} contains multiple samplerates: {sorted(sample_rate_values)}")
    sample_rate_hz = next(iter(sample_rate_values))

    n_time = int(round(duration_s * sample_rate_hz)) + 1
    stride = int(round(stride_s * sample_rate_hz))

    samples: list[Sample] = []
    for m_idx, m in enumerate(measurements):
        voltage, current, displacement_m, velocity_m_s = _asdf_measurement_to_signals(m)
        for seg_idx, (start, u) in enumerate(_windows_1d(voltage, n_time, stride, include_tail)):
            x = _pad_or_trim_1d(displacement_m[start:], n_time)
            v = _pad_or_trim_1d(velocity_m_s[start:], n_time)
            i = _pad_or_trim_1d(current[start:], n_time)
            states = np.stack([i, x, v], axis=1).astype(np.float64, copy=False)
            attrs = {
                "sample_idx": len(samples) + 1,
                "seed": int(m.get("seed", 0)),
                "source_file": str(asdf_path),
                "measurement_index": int(m_idx),
                "segment_index": int(seg_idx),
                "start_index": int(start),
            }
            for k, v in m.items():
                if k in {"input_data", "input_data_var", "output_data"}:
                    continue
                attrs[k] = v
            samples.append(
                Sample(
                    input_signal=_pad_or_trim_1d(u, n_time),
                    states=states,
                    attrs=attrs,
                )
            )

    return sample_rate_hz, samples, root_attrs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert ExpD MATLAB pinknoise and processed ASDF files to Loudspeaker.py-style HDF5.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    expd = subparsers.add_parser("expd", help="Convert DTU ExpD MATLAB recording.")
    expd.add_argument(
        "--input",
        type=Path,
        default=Path("dtu_34871-NL-Transducers/ExpD/pinknoise_5Hz-2000Hz_8Vrms.mat"),
        help="Path to the ExpD MATLAB file.",
    )
    expd.add_argument(
        "--output",
        type=Path,
        default=Path("hdf5_datasets/ExpD/pinknoise_5Hz-2000Hz_8Vrms_timeseries_dataset.h5"),
        help="Output HDF5 path.",
    )
    expd.add_argument("--group", default="nonlinear_loudspeaker", help="HDF5 group name.")
    expd.add_argument("--duration-seconds", type=float, default=5.0, help="Per-sample duration.")
    expd.add_argument("--stride-seconds", type=float, default=5.0, help="Stride between windows.")
    expd.add_argument(
        "--include-tail",
        action="store_true",
        help="Include the final partial window by padding with edge values.",
    )
    expd.add_argument("--force", action="store_true", help="Overwrite the output file if it exists.")

    asdf_cmd = subparsers.add_parser("asdf", help="Convert processed speaker ASDF file.")
    asdf_cmd.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the processed .asdf file.",
    )
    asdf_cmd.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output HDF5 path.",
    )
    asdf_cmd.add_argument("--group", default="nonlinear_loudspeaker", help="HDF5 group name.")
    asdf_cmd.add_argument("--duration-seconds", type=float, default=5.0, help="Per-sample duration.")
    asdf_cmd.add_argument("--stride-seconds", type=float, default=5.0, help="Stride between windows.")
    asdf_cmd.add_argument(
        "--include-tail",
        action="store_true",
        help="Include the final partial window by padding with edge values.",
    )
    asdf_cmd.add_argument(
        "--partition-variants",
        action="store_true",
        help="Group samples by signal/duration/seed/vrms into nested subgroups.",
    )
    asdf_cmd.add_argument("--force", action="store_true", help="Overwrite the output file if it exists.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path: Path = args.output.resolve()
    if output_path.exists() and not args.force:
        raise FileExistsError(f"{output_path} exists. Pass --force to overwrite.")

    if args.command == "expd":
        input_path = args.input.resolve()
        if not input_path.exists():
            raise FileNotFoundError(input_path)
        sample_rate_hz, samples, root_attrs = _expd_samples(
            input_path,
            duration_s=args.duration_seconds,
            stride_s=args.stride_seconds,
            include_tail=args.include_tail,
        )
    else:
        input_path = args.input.resolve()
        if not input_path.exists():
            raise FileNotFoundError(input_path)
        if args.partition_variants:
            measurements, root_attrs = _asdf_measurements(input_path)
            _write_partitioned_asdf_h5(
                output_path,
                group_name=args.group,
                measurements=measurements,
                root_attrs=root_attrs,
                source_path=input_path,
            )
            print(f"Wrote {output_path} with partitioned variants from {len(measurements)} measurements.")
            return
        sample_rate_hz, samples, root_attrs = _asdf_samples(
            input_path,
            duration_s=args.duration_seconds,
            stride_s=args.stride_seconds,
            include_tail=args.include_tail,
        )

    time = _time_vector(sample_rate_hz, args.duration_seconds)
    _write_h5(
        output_path,
        time=time,
        group_name=args.group,
        samples=samples,
        root_attrs=root_attrs,
    )
    print(f"Wrote {output_path} with {len(samples)} samples at {sample_rate_hz} Hz.")


if __name__ == "__main__":
    main()
