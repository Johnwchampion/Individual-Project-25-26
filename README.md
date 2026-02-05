# Measuring and Steering Faithfulness in Mixture-of-Experts Language Models via Expert Deactivation

## Overview

This repository contains the research codebase for my Individual Project (2025/26).

The project follows the direction of **“Steering MoE via Expert Deactivation”**, focusing on
**faithfulness evaluation and steering** in mixture-of-experts (MoE) language models. The central
idea is to:

- work with a **grounded QA dataset** containing questions, supporting documents, and answers,
- measure how **expert routing** in a MoE model (e.g. DeepSeek-V2-Lite) changes when evidence
  documents are present vs absent, and
- use this to identify and later steer experts whose activation patterns are tied to faithful vs
  unfaithful (hallucinated) behaviour.

The project is now significantly simpler than the initial design: it does **not** study multiple
reasoning regimes, explicit reasoning traces, or expert “reasoning profiles”. Instead, it focuses
on **document-grounded faithfulness** and **expert deactivation for steering**.

---

## High-Level Goals

- Build a clean pipeline from a **known QA dataset with supporting documents** to a format suitable
  for DeepSeek-V2-Lite.
- Log **expert routing traces** for different input conditions (with and without supporting
  documents).
- Compute a **risk-difference style metric** per expert that captures how strongly its activation
  depends on the presence of the supporting document.
- Identify experts that are most sensitive to document presence and thus promising candidates for
  later **expert deactivation / steering**.
- Evaluate how steering these experts affects **faithfulness** (groundedness to the supplied
  documents) and **task performance**.

---

# Stage 1: Data Pipeline and Expert Routing Analysis

Stage 1 replaces the earlier “reasoning regime” work. It has two main components:

1. **Data preparation and formatting for DeepSeek-V2-Lite**
2. **Routing trace extraction and expert risk-difference ranking**

Both components are designed to mirror, in a simplified form, the faithfulness analysis pipeline
used in the “Steering MoE via Expert Deactivation” paper.

---

## 1. Data Preparation and Formatting

### Objective

Construct a dataset of **(question, supporting document(s), answer)** triples from a **known
grounded QA benchmark** and format it into prompts consumable by DeepSeek-V2-Lite.

### Pipeline

- **Dataset selection**  
  Use an existing QA dataset where each question is paired with one or more supporting passages and
  a reference answer (e.g. open-domain QA or multi-hop QA). The specific dataset can be swapped
  without changing the rest of the pipeline.

- **Canonical schema**  
  Convert raw dataset entries into a unified internal format, for example:
  - `question`: natural language question
  - `documents`: list of supporting passages / snippets
  - `answer`: gold/reference answer text

- **Prompt construction**  
  For each example, build at least two prompt variants:
  - **No-document condition (Q-only)**: the model sees only the question.
  - **Document condition (Q+Doc)**: the model sees the question together with one or more supporting
    passages, formatted in an instruction style that clearly separates context from the question.

- **Serialisation**  
  Save the resulting prompts (and pointers to the original QA metadata) into JSONL files under
  `stage1/data/`, ready to be fed into the MoE model inference pipeline.

The output of this component is a clean, model-ready dataset that systematically contrasts
**question-only** vs **question-plus-document** inputs for the same underlying QA examples.

---

## 2. Routing Trace Extraction and Expert Risk Difference

### Objective

Analyse how the **Mixture-of-Experts router** in DeepSeek-V2-Lite responds to the presence or
absence of supporting documents, and rank experts by a **risk-difference style measure** that
captures this sensitivity.

### Routing Trace Logging

- Run DeepSeek-V2-Lite on the prepared prompts under two conditions:
  - **Q-only** inputs (no supporting document)
  - **Q+Doc** inputs (with supporting document)

- For each forward pass, log the **token-level routing decisions**, including (depending on what
  the model exposes):
  - gate probabilities for each expert,
  - which experts were selected (top-k routing),
  - per-layer and per-token expert assignments.

- Aggregate these token-level signals into **sequence-level expert activation summaries** for each
  example and condition, such as:
  - total/average gate mass per expert,
  - proportion of tokens routed to each expert.

### Risk Difference per Expert

Using the aggregated routing summaries, compute for each expert a **risk-difference style score**
that captures how much its activation changes when a supporting document is present:

- For each example and expert, measure an activation statistic under:
  - `A_with_doc`  – activation when the supporting document is provided
  - `A_without_doc` – activation when the document is withheld

- Define a **risk-difference style metric**, for example:
  - `RD_expert = E[A_with_doc] - E[A_without_doc]` (averaged over examples)

- Rank experts by `RD_expert` to identify:
  - experts that **strongly increase** activation when a document is present (document-sensitive),
  - experts that are largely **insensitive** to document presence.

This ranking is the core Stage 1 output used to support later **expert deactivation / steering
experiments**: high-risk-difference experts are natural candidates for intervention.

---

## Faithfulness Evaluation and Steering (Beyond Stage 1)

While Stage 1 focuses on data preparation and routing analysis, the broader project goal is to
adapt the “Steering MoE via Expert Deactivation” methodology to evaluate and improve faithfulness:

- Use the Stage 1 expert rankings to select experts for **deactivation or down-weighting**.
- Compare model outputs with and without these experts active under Q-only and Q+Doc conditions.
- Measure how steering affects:
  - **faithfulness**: alignment of model answers with the provided supporting documents,
  - **hallucination rate**: tendency to produce unsupported or contradicted statements,
  - **task performance**: accuracy/utility on the QA task.

These later stages will be built on top of the Stage 1 outputs but are intentionally left flexible
to allow iteration on evaluation metrics and steering strategies.

---

## Repository Layout (High-Level)

The repository is organised as follows (subject to change as the project evolves):

- `src/` – Core scripts for data preparation, model interaction, and analysis.
- `stage1/` – Assets and scripts specific to Stage 1:
  - `data/` – Processed QA + document prompts (JSONL) and related artefacts.
  - `analysis/` – Intermediate routing summaries, expert rankings, and notebooks.
  - `run/` – Logs, raw routing traces, and model outputs from DeepSeek-V2-Lite.
- `docs/` – Project notes and extended documentation.
- `tests/` – Unit tests and simple sanity checks for the pipeline.

As the project develops, these folders will be refined, but the conceptual focus will remain on:
**faithfulness evaluation and expert-based steering in MoE models**, following the SteerMoE
approach.

