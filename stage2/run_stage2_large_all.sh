#!/bin/bash
#SBATCH --job-name=stage2_all_large
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/sc23jc3/stage2_all_large_%j.log

source ~/envs/deepseek/bin/activate

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Override with: sbatch --export=ALL,N=300 run_stage2_large_all.sh
N=${N:-300}
SOFT_STRENGTH=${SOFT_STRENGTH:-0.5}
CANDIDATE_N=${CANDIDATE_N:-3}

cd /users/sc23jc3/projects/Individual-Project-25-26/stage2

python -u src/run_stage2.py \
  --tasks safety_safe safety_unsafe faith_cf faith_un faith_mc fluency \
  --conditions baseline hard soft \
  --n "$N" \
  --candidate_n "$CANDIDATE_N" \
  --soft_strength "$SOFT_STRENGTH" \
  --results_dir /users/sc23jc3/projects/Individual-Project-25-26/stage2/results \
  --skip_done
