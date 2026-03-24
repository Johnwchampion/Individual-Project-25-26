#!/bin/bash
#SBATCH --job-name=stage2_safety
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/sc23jc3/stage2_results/safety_%j.log

source ~/envs/deepseek/bin/activate

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

cd /users/sc23jc3/projects/Individual-Project-25-26/stage2

python -u src/run_stage2.py \
    --tasks safety_safe safety_unsafe \
    --conditions baseline hard soft \
    --n 100 \
    --soft_strength 0.5
