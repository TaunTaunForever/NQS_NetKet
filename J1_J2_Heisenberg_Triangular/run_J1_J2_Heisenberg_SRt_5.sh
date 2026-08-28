#!/bin/bash
#SBATCH --account=rrg-sorensen
#SBATCH --nodes=1
#SBATCH --export=ALL,DISABLE_DCGM=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4000
#SBATCH --gpus=1          # here you should insert the total number of gpus per node
#SBATCH --time=6:00:00

# Creating virtual environment and installing NetKet+dependencies
source ~/ENV/bin/activate
nvidia-smi

# Tell Jax that we want to use GPUs. THis is generally not needed but can't hurt
export JAX_PLATFORM_NAME=gpu

# Activate virtual environment and run NetKet script
mpirun -n 1 python3 J1_J2_Heisenberg_SRt_5.py
