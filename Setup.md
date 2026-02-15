# Aire Setup Summary

We use the Aire HPC cluster for all DeepSeek experiments.

## Access
- Connected via SSH.
- Used VS Code Remote-SSH to edit files directly on Aire.
- No rsync needed; code lives on the cluster.

## Project Location
- Project directory: /users/sc23jc3/projects/Individual-Project-25-26/
- Stage 1 code: stage1/src/

## Python Environment
- Created using Python venv:
  python -m venv ~/envs/deepseek
- Activated with:
  source ~/envs/deepseek/bin/activate
- Installed torch + transformers stack via pip.

## GPU Usage
- Interactive testing via:
  srun --partition=gpu --gres=gpu:1 --mem=40G --time=02:00:00 --pty bash
- All inference runs on GPU nodes (not login nodes).

## Model Caching
- Hugging Face cache stored in:
  /scratch/sc23jc3/cache
- Absolute path required (leading `/`).
- Using relative paths caused large model downloads into home directory.

## Model
- deepseek-ai/DeepSeek-V2-Lite-Chat
- Loaded with transformers and trust_remote_code=True
- Chat formatting handled via tokenizer.apply_chat_template()

This setup supports Stage 1 expert routing profiling experiments.

