#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")"
sbatch run_vit_srt_j2_sweep.sh
