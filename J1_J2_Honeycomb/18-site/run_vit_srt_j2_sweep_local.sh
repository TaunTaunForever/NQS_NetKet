#!/bin/bash

set -euo pipefail

if [[ -n "${J1J2_SWEEP_VALUES:-}" ]]; then
  # Accept a whitespace-separated override list, for example:
  # J1J2_SWEEP_VALUES="0.05 0.15 0.25"
  read -r -a J2_VALUES <<< "${J1J2_SWEEP_VALUES}"
else
  J2_VALUES=(
    0.00
    0.10
    0.20
    0.22
    0.24
    0.25
    0.30
    0.35
    0.40
    0.50
  )
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [[ -n "${J1J2_VENV:-}" ]]; then
  # Allows overriding the Python environment externally.
  # shellcheck disable=SC1090
  source "${J1J2_VENV}/bin/activate"
elif [[ -f "${PROJECT_ROOT}/NetKet_Updated_venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "${PROJECT_ROOT}/NetKet_Updated_venv/bin/activate"
fi

if [[ -z "${J1J2_PLATFORM:-}" ]]; then
  export JAX_PLATFORM_NAME=gpu
else
  export JAX_PLATFORM_NAME="${J1J2_PLATFORM}"
fi

export J1J2_J1="${J1J2_J1:-1.0}"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"

cd "${SCRIPT_DIR}"
echo "Running local 18-site J1-J2 SRt sweep with J1=${J1J2_J1} on ${JAX_PLATFORM_NAME}"

for j2 in "${J2_VALUES[@]}"; do
  export J1J2_J2="${j2}"
  echo
  echo "=== J2=${J1J2_J2} ==="
  python3 vit_srt.py
done
