#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Joining archived datasets..."
bash "${repo_root}/scripts/join_data.sh"

echo "Converting MATLAB files to NumPy archives..."
python "${repo_root}/scripts/convert_pinknoise_to_npz.py" "$@"
