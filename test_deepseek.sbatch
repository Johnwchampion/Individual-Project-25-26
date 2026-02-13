#!/bin/bash
#SBATCH --job-name=deepseek_test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/deepseek_test_%j.out
#SBATCH --error=logs/deepseek_test_%j.err

module load miniforge
conda activate deepseek-env

cd ~/projects/Individual-Project-25-26

nvidia-smi

python stage1/src/test_deepseek.py

