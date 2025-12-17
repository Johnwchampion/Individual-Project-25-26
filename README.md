# Measuring faithfulness and performance in mixture-of-experts language models through expert misrouting

## Overview
This repository contains the initial scaffold for my Individual Project (2025/26).  
The project will explore **how mixture-of-experts (MoE) language models behave when their routing decisions are intentionally altered**. The aim is to study whether models remain **faithful to their internal computation** when producing explanations, and how performance changes under controlled misrouting.

At this stage, the repository only includes a basic folder structure.  
No code, experiments, or models have been added yet.

## Goals (High-Level)
- Investigate how MoE router decisions influence model predictions.  
- Identify experts that contribute to specific reasoning behaviours.  
- Apply controlled **expert misrouting** to probe the model’s internal computation.  
- Measure **faithfulness**, defined as the alignment between internal causal pathways and external explanations.  
- Compare faithfulness with task performance across different intervention types.

These goals will be refined as the project develops.

# Stage 1: Sequence-Level Expert Routing Profiling

## Purpose and Scope

In mixture-of-experts (MoE) language models, routing decisions are made at the token level: for each token, the router selects one expert to process it. However, it is not meaningful to interpret individual tokens as being “easy” or “difficult.” Difficulty is a property of the entire prompt or task, not of isolated tokens.

The goal of Stage 1 is therefore **not** to analyse token-level difficulty, but to **aggregate token-level routing decisions into prompt-level summaries**. These summaries describe how computation is distributed across experts after the full prompt has been processed. This enables comparison of expert usage patterns across prompts of different difficulty levels.

The central hypothesis at this stage is:

> Harder prompts induce a different overall expert routing distribution than easier prompts.

All analysis in this stage is prompt-level and distributional.

---

## Token-Level Routing Signal

Consider a prompt \( x \) consisting of \( T \) tokens.  
In a Switch-style MoE layer with \( E \) experts and top-1 routing, the router selects exactly one expert for each token.

Let  
\[
e(t) \in \{1, \dots, E\}
\]
denote the expert selected for token \( t \).

These token-level routing decisions are treated as raw computational signals and are **not** interpreted as indicators of token difficulty.

---

## Sequence-Level Routing Distribution

For each prompt \( x \), we define a **sequence-level routing distribution**
\[
u(x) \in \mathbb{R}^E
\]
whose components represent the fraction of tokens routed to each expert.

For expert \( i \), this is computed as:
\[
u_i(x) = \frac{1}{T} \sum_{t=1}^{T} \mathbf{1}[e(t) = i]
\]

where \( \mathbf{1}[\cdot] \) is the indicator function.

**Interpretation**

- \( u_i(x) \) is the fraction of tokens in prompt \( x \) routed to expert \( i \)
- \( u(x) \) can be interpreted as the *final routing distribution* for the prompt
- This distribution summarises how the model allocated its computation across experts for the entire input

This aggregation avoids per-token interpretation while producing a single, interpretable routing signature per prompt.

---

## Optional: Restricting to the Answer / Reasoning Region

Many tokens correspond to reading or formatting rather than reasoning. To better capture computation associated with problem-solving, routing aggregation can optionally be restricted to a subset of tokens.

Let  
\[
S(x) \subseteq \{1, \dots, T\}
\]
denote a selected token subset, such as:
- tokens after a delimiter like `"Answer:"`
- tokens generated during the model’s response
- the final \( M \) tokens of the sequence

The restricted routing distribution is then:
\[
u_i(x) = \frac{1}{|S(x)|} \sum_{t \in S(x)} \mathbf{1}[e(t) = i]
\]

This produces a routing signature more closely aligned with the model’s reasoning behaviour rather than prompt parsing.

---

## Aggregation by Difficulty Level

Prompts are grouped into discrete difficulty levels \( k \in \{1, \dots, K\} \) (e.g. easy, medium, hard).

For each difficulty level \( k \), we compute the mean routing distribution across all prompts in that group:
\[
\bar{u}_i(k) = \frac{1}{|C_k|} \sum_{x \in C_k} u_i(x)
\]
where \( C_k \) denotes the set of prompts at difficulty level \( k \).

This yields one average routing distribution per difficulty level, enabling comparison of expert usage patterns as task difficulty increases.

---

## Measuring Difficulty-Related Expert Trends

Some experts may be heavily used across all prompts regardless of difficulty. To identify experts whose usage changes *systematically* with difficulty, we compute a trend score for each expert:
\[
\Delta_i = \sum_{k=1}^{K} w_k \, \bar{u}_i(k)
\quad \text{subject to} \quad
\sum_{k=1}^{K} w_k = 0
\]

The zero-sum constraint ensures that:
- experts used uniformly across difficulty levels have scores near zero
- positive values of \( \Delta_i \) indicate increasing usage with difficulty
- negative values indicate decreasing usage with difficulty

Experts can then be ranked by \( |\Delta_i| \) to identify those most sensitive to task difficulty.

---

## Outcome of Stage 1

Stage 1 produces:

1. A sequence-level routing distribution \( u(x) \) for each prompt  
2. Average routing distributions \( \bar{u}(k) \) for each difficulty level  
3. A ranked list of experts whose usage varies most consistently with difficulty  

These results are **descriptive, not causal**. They provide a principled, low-cost profiling method for selecting candidate experts for intervention.

---

## Role in the Overall Project

The experts identified in Stage 1 are used as candidates for controlled routing interventions in Stage 2. By deliberately misrouting or suppressing these experts, the project studies how perturbations in internal routing affect:

- task performance  
- output stability  
- faithfulness of generated explanations  

Stage 1 therefore serves as an exploratory profiling step that grounds later causal analysis in observed routing behaviour.

