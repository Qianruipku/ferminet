#!/bin/bash

#SBATCH -J ferminet
#SBATCH -o 1.out
#SBATCH -e 1.err
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --time=1-00:00:00

# source ~/pythonenv/jaxgpu/bin/activate

export NVIDIA_TF32_OVERRIDE=0
export JAX_DEFAULT_MATMUL_PRECISION=highest
export JAX_ENABLE_X64=0

PORT=1345
IP_ADDR=$(ifconfig 2> /dev/null | awk '$1 == "inet" {print $2}' | head -n 2 | tail -n 1)


srun --nodes=2 \
     --gres=gpu:4 \
     --export=ALL \
     python -u ch4.py --server_addr="$IP_ADDR:$PORT"