# Measuring Faithfulness and Performance in Mixture-of-Experts Language Models Through Expert Misrouting

## Overview

This repository contains the research codebase for my Individual Project (2025/26).  
The project investigates **faithfulness and performance in mixture-of-experts (MoE) language models** by studying how model behaviour changes when **expert routing decisions are intentionally perturbed**.

Mixture-of-experts models dynamically route tokens to different subnetworks (“experts”) during generation. While this architecture improves efficiency and capacity, it also introduces a potential gap between **internal computation** (expert routing) and **external explanations** (e.g. chain-of-thought reasoning). This project explores whether models remain faithful to their internal causal pathways when producing explanations, and how performance and stability change under controlled routing interventions.

The project is organised into multiple stages.  
This repository currently implements **Stage 1: Sequence-Level Expert Routing Profiling**, which provides the descriptive foundation for later causal analysis.

---

## High-Level Goals

- Characterise expert routing behaviour in MoE language models  
- Identify experts that are systematically associated with reasoning-style generation  
- Apply controlled expert misrouting or suppression to probe internal computation  
- Measure faithfulness as alignment between internal routing behaviour and external explanations  
- Compare faithfulness with task performance and output stability under intervention  

---

# Stage 1: Sequence-Level Expert Routing Profiling

## Purpose

In MoE language models, routing decisions are made at the **token level**, with a router selecting an expert for each token. However, reasoning is not meaningfully defined at the level of individual tokens. Instead, reasoning is treated as a property of the **entire generation process**, shaped by task structure and prompting strategy.

Stage 1 therefore does **not** attempt to identify which tokens or experts “perform reasoning”.  
Instead, it aggregates token-level routing decisions into **sequence-level summaries** that describe how computation is distributed across experts over a full prompt and generation.

Stage 1 is **descriptive rather than causal**. Its role is to identify systematic patterns in expert usage under different reasoning-elicitation conditions, without assuming that any condition corresponds to greater or lesser internal reasoning depth.

---

## Reasoning-Elicitation Regimes

Each base prompt is evaluated under four controlled prompting regimes, each corresponding to a well-established empirical paradigm known to alter how reasoning is expressed during generation:

- **R0 – Zero-Shot Direct**  
  The model is prompted to produce a direct answer with no explicit reasoning cues. All intermediate computation remains latent. This serves as the baseline.

- **R1 – Zero-Shot Chain-of-Thought**  
  A minimal cue (e.g. “Let’s think step by step”) is appended to trigger an explicit reasoning trace without providing examples.

- **R2 – Few-Shot Chain-of-Thought with Persona**  
  The prompt includes several solved exemplars with intermediate reasoning steps, combined with a persona-style instruction (e.g. “Act as a logical auditor”), introducing both structural templates and semantic guidance.

- **R3 – Structured Self-Consistency**  
  Multiple independent reasoning trajectories are sampled for the same prompt. Routing behaviour is analysed per trajectory and then averaged, allowing identification of experts that persist across redundant reasoning paths.

These regimes are treated as **distinct empirical conditions**, not as a linear scale of reasoning strength.

---

## Prompt Design

Prompts are **domain-agnostic** and selected using **structural criteria** rather than semantic categories (e.g. “math” or “logic”), to avoid domain-driven routing confounds.

Each prompt is designed to require:
- recurring entity definitions,
- intermediate dependencies between steps,
- implicit state tracking across the task.

The same base prompts are used across all reasoning regimes to ensure that observed routing differences reflect changes in reasoning paradigm rather than surface-level formatting effects.

---

## Sequence-Level Routing Profiles

During inference, the model’s token-level expert routing decisions are logged.  
These token-level signals are then **aggregated across the full sequence** to produce a single routing profile per prompt, representing how often each expert was used during generation.

Routing profiles are computed at the sequence level rather than the token level to avoid attributing reasoning significance to individual tokens.

---

## Aggregation by Reasoning Regime

For each reasoning regime, routing profiles are averaged across all prompts to produce a **regime-level expert usage profile**.

For the self-consistency regime (R3), routing profiles are computed separately for each generated reasoning trajectory and then averaged, enabling analysis of expert persistence across multiple independent reasoning paths.

---

## Identifying Recurring Experts

Stage 1 identifies experts that:
- increase in usage relative to the zero-shot baseline,
- do so **consistently across multiple reasoning regimes**, and
- remain active across redundant reasoning trajectories under self-consistency.

These experts are considered **recurring experts** and are ranked using recurrence-based metrics. Importantly, these rankings are descriptive and do not imply causal importance on their own.

---

## Outputs of Stage 1

Stage 1 produces:

- A sequence-level routing profile for each prompt instance  
- Average routing

