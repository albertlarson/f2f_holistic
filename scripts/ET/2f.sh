#!/bin/bash -l
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1
#SBATCH --job-name=1
#SBATCH --mem=130000
#SBATCH -o ../data/GLDAS/clipped/yukon/2g_f2f_holistic-%j.out  # %j = job ID

module purge

eval "$(conda shell.bash hook)"

conda info --envs

conda activate /work/albertl_uri_edu/.conda/envs/f2f_2

python -u 2g_clip_yukonaspy.py  > ../data/GLDAS/clipped/yukon/log.out 