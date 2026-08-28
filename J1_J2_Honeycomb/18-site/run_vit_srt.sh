#!/bin/bash
#SBATCH --account=rrg-sorensen
#SBATCH --nodes=1
#SBATCH --export=ALL,DISABLE_DCGM=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4000
#SBATCH --gpus=1
#SBATCH --time=08:00:00

source ~/ENV/bin/activate
nvidia-smi

export JAX_PLATFORM_NAME="${JAX_PLATFORM_NAME:-gpu}"

cd "$(dirname "$0")"
mpirun -n 1 python3 vit_srt.py
