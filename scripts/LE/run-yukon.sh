#!/bin/bash -l
#SBATCH --partition=gpu-long
#SBATCH --time=48:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1
#SBATCH --job-name=f2f_yuk
#SBATCH --mem=64000
#SBATCH -o /work/albertl_uri_edu/f2f_holistic/data/results/outputfiles/job-yukon-UNITY-%j.out  # %j = job ID

module purge
eval "$(conda shell.bash hook)"
conda info --envs
conda activate /work/albertl_uri_edu/.conda/envs/f2f_2
python -u run-yukon.py  > /work/albertl_uri_edu/f2f_holistic/data/results/outputfiles/run-yukon.out 