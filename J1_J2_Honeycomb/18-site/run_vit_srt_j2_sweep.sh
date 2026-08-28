#!/bin/bash
#SBATCH --account=rrg-sorensen
#SBATCH --nodes=1
#SBATCH --export=ALL,DISABLE_DCGM=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4000
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --array=0-3

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
export J1J2_EMBED_DIM="${J1J2_EMBED_DIM:-24}"
export J1J2_SAMPLER="${J1J2_SAMPLER:-local}"
export J1J2_SAMPLER_REFINE="${J1J2_SAMPLER_REFINE:-local}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
mkdir -p "${MPLCONFIGDIR}"

cd "$(dirname "$0")"
echo "Running 18-site J1-J2 SRt sweep with J1=${J1J2_J1}, J2=${J1J2_J2}"
echo "Embed dim=${J1J2_EMBED_DIM}, sampler=${J1J2_SAMPLER}, refine sampler=${J1J2_SAMPLER_REFINE}"
mpirun -n 1 python3 vit_srt.py
