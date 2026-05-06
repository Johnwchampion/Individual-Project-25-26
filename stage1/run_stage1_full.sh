#!/bin/bash
#SBATCH --job-name=stage1_full
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/sc23jc3/stage1_full_%j.log

source ~/envs/deepseek/bin/activate

cd /users/sc23jc3/projects/Individual-Project-25-26

echo "=== Stage 1 full rerun ==="
echo "Started at: $(date)"

# --- Step 1: prepare faithfulness data (FaithEval-CF, CPU only) ---
if [ -f /scratch/sc23jc3/faithcf_prepared/faithcf_chat_formatted.jsonl ]; then
    echo "[1/3] FaithEval-CF data already exists, skipping."
else
    echo "[1/3] Preparing FaithEval-Counterfactual data..."
    python stage1/prep/prepare_faithdata.py
    echo "Done at: $(date)"
fi

# --- Step 2: prepare safety data (AdvBench, requires GPU for generation) ---
if [ -f /scratch/sc23jc3/advbench_prepared/advbench_safety_pairs.jsonl ]; then
    echo "[2/3] AdvBench safety pairs already exist, skipping."
else
    echo "[2/3] Preparing AdvBench safety pairs (200 pairs, GPU generation)..."
    python stage1/prep/prepare_safedata.py
    echo "Done at: $(date)"
fi

# --- Step 3: run Stage 1 expert profiling on both axes ---
echo "[3/3] Running Stage 1 expert profiling (faith + safety)..."

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

python -u stage1/src/run_stage1.py --mode both
echo "Done at: $(date)"

echo "=== Stage 1 complete ==="
echo "RD outputs:"
echo "  /scratch/sc23jc3/results/rd_faithfulness.json"
echo "  /scratch/sc23jc3/results/rd_faithfulness_logits.json"
echo "  /scratch/sc23jc3/results/rd_safety.json"
echo "  /scratch/sc23jc3/results/rd_safety_logits.json"
