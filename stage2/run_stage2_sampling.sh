#!/bin/bash
#SBATCH --job-name=stage2_sampling
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --array=1-5
#SBATCH --output=/scratch/sc23jc3/stage2_sampling_%A_%a.log

source ~/envs/deepseek/bin/activate

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

SEED=${SLURM_ARRAY_TASK_ID:-1}
N=${N:-500}
SOFT_STRENGTH=${SOFT_STRENGTH:-0.5}
CANDIDATE_N=${CANDIDATE_N:-3}
RESULTS_DIR=${RESULTS_DIR:-/users/sc23jc3/projects/Individual-Project-25-26/stage2/results}

cd /users/sc23jc3/projects/Individual-Project-25-26/stage2

echo "Starting Stage 2 sampling: SEED=$SEED  N=$N  CANDIDATE_N=$CANDIDATE_N  SOFT_STRENGTH=$SOFT_STRENGTH"
echo "Results dir: $RESULTS_DIR"
echo "Started at: $(date)"

# Stochastic sampling — safety tasks, independent decoding seed per array task
python -u src/run_stage2.py \
  --tasks safety_safe safety_unsafe \
  --conditions baseline hard soft \
  --n "$N" \
  --candidate_n "$CANDIDATE_N" \
  --soft_strength "$SOFT_STRENGTH" \
  --results_dir "$RESULTS_DIR" \
  --seed "$SEED" \
  --skip_done

# Random candidate ablation — same array task ID used as candidate randomisation seed
python -u src/run_stage2.py \
  --tasks safety_safe safety_unsafe \
  --conditions random \
  --n "$N" \
  --candidate_n "$CANDIDATE_N" \
  --results_dir "$RESULTS_DIR" \
  --random_seed "$SEED" \
  --skip_done

echo "Finished at: $(date)"
