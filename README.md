# Loudspeaker Datasets

Curated measurements and utilities for working with loudspeaker data gathered
from DTU experiments and related sources.

## Requirements

- [pixi](https://pixi.sh) for dependable Python environments.
- `tar` and `zstd` when rebuilding archives from the `parts/` directory.

The pixi environment installs Python, NumPy, and SciPy which are required for
the conversion utilities.

## Preparing the data

The repository contains split archives of the raw measurements. Combine them
and convert the DTU ExpD MATLAB recording into an `.npz` archive with:

```bash
pixi run prep_data
```

The `prep_data` task runs `scripts/join_data.sh` to join archives and then
executes `scripts/convert_pinknoise_to_npz.py` to create
`numpy_datasets/ExpD/pinknoise_5Hz-2000Hz_8Vrms.npz` together with its metadata.
If you only need the conversion step or want to tweak arguments, run the script
directly:

```bash
python scripts/convert_pinknoise_to_npz.py \
  --input dtu_34871-NL-Transducers/ExpD/pinknoise_5Hz-2000Hz_8Vrms.mat \
  --output-dir numpy_datasets \
  --dataset-name pinknoise_5Hz-2000Hz_8Vrms \
  --experiment ExpD
```

Add `--force` when overwriting an existing archive.

To convert *every* ExpD MATLAB file in one go (once the archives have been
extracted), use the dedicated pixi task:

```bash
pixi run convert_expd
```

This command iterates over `dtu_34871-NL-Transducers/ExpD/*.mat` and writes
one `.npz`/`.json` pair per recording under `numpy_datasets/ExpD/`, which is
git-ignored so every contributor builds their own local cache. Extra CLI flags
such as `--pattern` or `--force` can be passed directly to
`python scripts/convert_expd_to_npz.py`.

## Working with the `.npz` files

Each conversion produces a compressed `.npz` next to a JSON metadata file that
summarizes shapes, dtypes, and the source `.mat` file. Load and inspect the data
with NumPy:

```python
import json
import numpy as np
from pathlib import Path

npz_path = Path("numpy_datasets/ExpD/pinknoise_5Hz-2000Hz_8Vrms.npz")
data = np.load(npz_path, allow_pickle=False)
print("arrays:", data.files)  # discover available signals
drive_voltage = data["voltage"]
sample_rate = int(data["sample_rate"])

metadata = json.loads(npz_path.with_suffix(".json").read_text())
print(metadata["arrays"]["voltage"])
```

Use `data.files` and the metadata JSON to identify the canonical field names
before downstream processing.

## Loudspeaker dataset standard

To keep this repository consistent across future measurements:

- Store processed datasets under `numpy_datasets/<experiment>/<dataset>.npz`
  and accompany each archive with `<dataset>.json` metadata.
- Populate every `.npz` with a consistent set of canonical arrays:
  `voltage` (speaker terminal voltage in volts), `current` (amperes),
  `velocity` (cone velocity in m/s), `displacement` (cone displacement in m),
  and `sample_rate` (Hz). Optional additions such as
  `frequency_response`, `impulse_response`, or microphone position matrices
  should follow the `<quantity>_<unit>` naming pattern.
- Use SI units in the array names when practical (`spl_db`, `voltage_v`,
  `current_a`, `distance_m`) so downstream tooling can be unit-aware.
- Provide experiment-level context (excitation type, environment, gain staging,
  calibration constants) in the metadata JSON. The conversion script already
  lists shape and dtype information; extend the metadata structure instead of
  embedding ad-hoc notes in array names.
- When adding new MATLAB sources, prefer reshaping or renaming arrays inside
  the conversion script so the resulting `.npz` adheres to the field names
  above. That keeps downstream training/evaluation pipelines consistent
  regardless of the original measurement format.

Following this convention ensures every loudspeaker dataset delivers the same
core signals and descriptive metadata, even when their acquisition methods
differ.
