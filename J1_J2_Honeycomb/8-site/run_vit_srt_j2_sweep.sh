#!/bin/bash
#SBATCH --account=rrg-sorensen
#SBATCH --nodes=1
#SBATCH --export=ALL,DISABLE_DCGM=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4000
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --array=0-9

set -euo pipefail

J2_VALUES=(
  0.30
  0.35
  0.40
  0.50
)

source ~/ENV/bin/activate
nvidia-smi

export JAX_PLATFORM_NAME="${JAX_PLATFORM_NAME:-gpu}"
export J1J2_J1="${J1J2_J1:-1.0}"
export J1J2_J2="${J2_VALUES[$SLURM_ARRAY_TASK_ID]}"

cd "$(dirname "$0")"
echo "Running 8-site J1-J2 SRt sweep with J1=${J1J2_J1}, J2=${J1J2_J2}"
mpirun -n 1 python3 vit_srt.py
