# Measuring and Steering Faithfulness and Safety in Mixture-of-Experts Language Models via Expert Deactivation

## Overview

This repository contains the research codebase for my Individual Project (2025/26).

The project follows the direction of **"Steering MoE via Expert Deactivation"**, focusing on
**faithfulness and safety evaluation and steering** in mixture-of-experts (MoE) language models.

The central idea is to:

- construct controlled behavioural contrasts,
- measure how **expert routing** in a MoE model (DeepSeek-V2-Lite) changes across these regimes, and
- identify experts whose activation patterns are systematically associated with:
  - document-grounded vs ungrounded behaviour (faithfulness),
  - safe vs unsafe behaviour (safety alignment).

The project studies two orthogonal behavioural axes:

1. **Document-grounded faithfulness**
2. **Safety alignment (harmful compliance vs refusal)**

The focus is on **behavioural contrasts and routing sensitivity**, following a simplified and controlled adaptation of the SteerMoE methodology.

---

# High-Level Goals

- Build clean pipelines for:
  - a **grounded QA dataset** (faithfulness axis),
  - a **safety contrast dataset** (unsafe vs safe completions).
- Log **expert routing traces** in DeepSeek-V2-Lite under controlled input conditions.
- Compute a **risk-difference style metric** per expert that captures behavioural sensitivity.
- Identify experts strongly associated with:
  - document-conditioned activation,
  - safety-aligned refusal behaviour.
- Evaluate how steering these experts affects:
  - faithfulness,
  - safety,
  - task performance.

---

# Stage 1: Behavioural Data Pipelines and Expert Routing Analysis

Stage 1 consists of two parallel pipelines:

1. **Faithfulness Pipeline** — document-sensitive routing analysis using SQuAD
2. **Safety Pipeline** — alignment-sensitive routing analysis using BeaverTails

Both pipelines feed into a shared routing analysis framework.

---

# Faithfulness Axis: Document-Grounded QA

## Dataset

- Source: `decodingchris/clean_squad_v2` (SQuAD v2)
- Prepared by: `stage1/prep/prepare_faithdata.py`
- Output: `/scratch/sc23jc3/squad_prepared/squad_chat_formatted.jsonl`

Each example is serialised into **two chat-formatted records**:
- `with_context` — system prompt + context document + question
- `no_context` — system prompt + question only

Records are paired by a shared base ID (`squad_XXXXXX_ctx` / `squad_XXXXXX_base`).

## Routing Analysis

- Orchestrated by: `stage1/src/run_stage1.py`
- Samples up to 500 pairs; runs teacher-forced forward passes for each condition.
- Uses `RouterTracer` to hook DeepSeek's MoE gate modules (`model.layers.X.mlp.gate`) and capture top-6 expert indices per layer per token.
- Routing is sliced to **question tokens only** using subsequence matching, isolating the signal from padding and system tokens.

## Risk Difference (Faithfulness)

For each expert `e` at each layer:

```
RD_faithfulness(e) = P(e activated | with_context) - P(e activated | no_context)
```

High positive RD → expert is document-sensitive (candidate for faithfulness steering).
Negative RD → expert is suppressed by context.

---

# Safety Axis: Unsafe vs Safe Behaviour

## Dataset

- Source: `innodatalabs/rt2-beavertails-simple` (BeaverTails)
- Prepared by: `stage1/prep/prepare_safedata.py`
- Output: `/scratch/sc23jc3/beavertails_prepared/beavertails_safety_pairs.jsonl`

The pipeline:
1. Filters for examples with a real user/assistant exchange (messages at index 3 and 4).
2. Keeps only examples labelled `expected == "unsafe"` (up to 40).
3. For each unsafe prompt:
   - Retains the **original harmful completion** as the `unsafe` record.
   - Generates a **synthetic safe refusal** using DeepSeek-V2-Lite-Chat with a safety-aligned system prompt as the `safe` record.

Safe responses are generated deterministically (greedy decoding, `temperature=0`) and are constrained to be:
- context-aware and explanatory,
- non-actionable,
- similar in length to the unsafe response.

Each output pair shares a base ID (`bt_XXXXXX_unsafe` / `bt_XXXXXX_safe`).

## Routing Measurement

For each pair:
- Run teacher-forced forward passes on both `unsafe` and `safe` assistant responses.
- Extract routing over **assistant response tokens only**.
- Optionally length-match using `K = min(L_safe, L_unsafe)` tokens.

## Risk Difference (Safety)

For each expert `e` at each layer:

```
RD_safety(e) = P(e activated | safe) - P(e activated | unsafe)
```

High positive RD → expert associated with refusal/alignment.
Negative RD → expert associated with harmful compliance.

---

# Steering via Expert Deactivation

After identifying high-risk-difference experts:

- Perform selective expert deactivation or down-weighting.
- Evaluate behavioural impact on:
  - grounded QA tasks (faithfulness),
  - unsafe prompts (safety compliance),
  - overall model utility.

Evaluation metrics may include:

- Answer correctness
- Groundedness to supplied documents
- Hallucination rate
- Safety refusal rate
- Task performance degradation

---

# Repository Layout

```
stage1/
  prep/
    prepare_faithdata.py    Build SQuAD faithfulness JSONL dataset
    prepare_safedata.py     Build BeaverTails safety-contrast JSONL dataset
    inspect_modules.py      Inspect model modules
    inspect_moe.py          Inspect MoE layer structure
    inspect_tokens.py       Token-level inspection utilities
    sanitycheck.py          Dataset/routing sanity checks
    test_deepseek.py        Quick model loading tests
  src/
    config.py               Paths and model name (faithfulness pipeline)
    dataset.py              ChatRecord / ChatPair dataclasses and JSONL loader
    model.py                load_model and generate utilities
    routing.py              RouterTracer — hooks MoE gates and records top-k experts
    visualize.py            accumulate_expert_counts, compute_rd, rank experts, plot
    run_stage1.py           Main Stage 1 orchestration script (faithfulness pipeline)
    inference_engine.py     Inference stub (in progress)
  demo/
    build_dataset.ipynb     Demo notebook: dataset construction
    switch_demo.ipynb       Demo notebook: routing switching
  plots/                    Saved routing instability plots
src/
  init.ipynb                Initial exploration notebook
  main.ipynb                Main notebook
docs/                       Extended project notes
tests/                      Unit tests and sanity checks
Setup.md                    AIRE HPC cluster setup notes
```

---

# Infrastructure

All experiments run on the **AIRE HPC cluster** (University of Leeds).

- Model: `deepseek-ai/DeepSeek-V2-Lite` / `deepseek-ai/DeepSeek-V2-Lite-Chat`
- GPU jobs: `srun --partition=gpu --gres=gpu:1 --mem=40G --time=02:00:00 --pty bash`
- HF cache: `/scratch/sc23jc3/cache`
- Data output: `/scratch/sc23jc3/`
- Python env: `~/envs/deepseek` (venv, PyTorch + transformers)

---

# Conceptual Summary

This project investigates whether specific experts in a Mixture-of-Experts language model:

- specialise in document-grounded reasoning,
- specialise in safety-aligned refusal behaviour,
- and can be selectively steered via expert deactivation.

By analysing routing sensitivity across controlled behavioural contrasts,
the project extends the SteerMoE framework to jointly study:

- **faithfulness**
- **safety**
- **expert-level interpretability and steering**
