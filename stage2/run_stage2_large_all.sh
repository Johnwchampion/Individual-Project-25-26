#!/bin/bash
#SBATCH --job-name=stage2_full
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/sc23jc3/stage2_full_%j.log

source ~/envs/deepseek/bin/activate

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

N=${N:-500}
SOFT_STRENGTH=${SOFT_STRENGTH:-0.5}
CANDIDATE_N=${CANDIDATE_N:-3}
RESULTS_DIR=${RESULTS_DIR:-/users/sc23jc3/projects/Individual-Project-25-26/stage2/results}

cd /users/sc23jc3/projects/Individual-Project-25-26/stage2

echo "Starting Stage 2 full run: N=$N  CANDIDATE_N=$CANDIDATE_N  SOFT_STRENGTH=$SOFT_STRENGTH"
echo "Results dir: $RESULTS_DIR"
echo "Started at: $(date)"

python -u src/run_stage2.py \
  --tasks safety_safe safety_unsafe faith_cf faith_un faith_mc \
  --conditions baseline hard soft \
  --n "$N" \
  --candidate_n "$CANDIDATE_N" \
  --soft_strength "$SOFT_STRENGTH" \
  --results_dir "$RESULTS_DIR" \
  --skip_done

echo "Finished at: $(date)"
