# Measuring and Steering Faithfulness and Safety in Mixture-of-Experts Language Models via Expert Deactivation

## Overview

This repository contains the research codebase for my Individual Project (2025/26).

The project follows the direction of **“Steering MoE via Expert Deactivation”**, focusing on
**faithfulness and safety evaluation and steering** in mixture-of-experts (MoE) language models.

The central idea is to:

- construct controlled behavioural contrasts,
- measure how **expert routing** in a MoE model (e.g. DeepSeek-V2-Lite) changes across these regimes, and
- identify experts whose activation patterns are systematically associated with:
  - document-grounded vs ungrounded behaviour (faithfulness),
  - safe vs unsafe behaviour (safety alignment).

The project studies two orthogonal behavioural axes:

1. **Document-grounded faithfulness**
2. **Safety alignment (harmful compliance vs refusal)**

Rather than modelling reasoning traces or expert “reasoning profiles”, the focus is on
**behavioural contrasts and routing sensitivity**, following a simplified and controlled
adaptation of the SteerMoE methodology.

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

1. **Faithfulness Pipeline (Document Sensitivity)**
2. **Safety Pipeline (Alignment Sensitivity)**

Both pipelines feed into a shared routing analysis framework.

---

# Faithfulness Axis: Document-Grounded QA

## Objective

Measure how expert routing changes when supporting evidence documents are present vs absent.

## Data Preparation

- Use a grounded QA benchmark with:
  - `question`
  - `supporting document(s)`
  - `reference answer`
- Construct two prompt variants per example:
  - **Q-only condition**
  - **Q+Doc condition**
- Serialize prompts into JSONL format for model consumption.

## Routing Trace Extraction

For each condition:

- Run DeepSeek-V2-Lite under teacher forcing.
- Log token-level routing decisions:
  - selected experts (top-k),
  - gate mass,
  - per-layer assignments.
- Aggregate to sequence-level expert statistics.

## Risk Difference (Faithfulness)

For each expert:

- Compute activation under:
  - `A_with_doc`
  - `A_without_doc`
- Define:

  `RD_faithfulness = E[A_with_doc] - E[A_without_doc]`

Experts with high positive RD are document-sensitive and candidates for steering.

---

# Safety Axis: Unsafe vs Safe Behaviour

## Objective

Measure how expert routing differs between:

- **Unsafe completions** (harmful compliance),
- **Safe refusals** (alignment behaviour).

## Dataset Construction

For each unsafe prompt:

- Use a dataset containing harmful or policy-violating completions.
- Construct paired examples:
  - `unsafe`: original harmful completion,
  - `safe`: model-generated refusal-style completion.

Safe responses are:

- context-aware,
- explanatory,
- length-comparable,
- non-actionable.

Both safe and unsafe responses are treated as fixed sequences for analysis.

## Routing Measurement

For each pair:

- Run teacher-forced forward passes on:
  - `User + Unsafe Assistant`
  - `User + Safe Assistant`
- Extract routing only over assistant response tokens.
- Optionally length-match at measurement time:
  - use `K = min(L_safe, L_unsafe)` tokens.

## Risk Difference (Safety)

For each expert:

- Compute:

  `RD_safety = E[A_safe] - E[A_unsafe]`

Experts with high positive RD_safety are associated with refusal/alignment behaviour.
Experts with negative RD_safety are associated with harmful compliance.

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
src/                Core scripts (model loading, routing hooks, metrics)
stage1/
  data/             Processed QA and safety datasets (JSONL)
  analysis/         Expert rankings, routing summaries, notebooks
  run/              Raw routing traces and logs
docs/               Project notes and extended documentation
tests/              Unit tests and sanity checks
```

---

# Conceptual Summary

This project investigates whether specific experts in a Mixture-of-Experts language model:

- specialize in document-grounded reasoning,
- specialize in safety-aligned refusal behaviour,
- and can be selectively steered via expert deactivation.

By analysing routing sensitivity across controlled behavioural contrasts,
the project extends the SteerMoE framework to jointly study:

- **faithfulness**
- **safety**
- **expert-level interpretability and steering**
