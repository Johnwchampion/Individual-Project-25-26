#!/usr/bin/env python3
"""
validate_steering.py  —  Mechanistic validation of ExpertSteerer hooks.

Runs N=50 prompts drawn from the actual experimental datasets in three
conditions (baseline / hard / soft) and records, for every targeted
(layer, expert) pair:

  Hard check  :  routing_rate == 0.0  (expert never selected post-hook)
  Soft check  :  routing_rate < baseline_rate (or both == 0)
                 actual_logit_shift  ≈  expected_shift  (strength × RD score)

Safety axis   : AdvBench prompts with forced harmful prefix  (mirrors safety_safe task)
Faithfulness  : FaithEval-Counterfactual prompts with context (mirrors faith_cf task)

Output  :  stage2/validation_result.json
"""

import sys
import os
import json
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "src"))
sys.path.insert(0, os.path.join(_HERE, "../stage1/src"))
sys.path.insert(0, os.path.join(_HERE, "../prep"))

from transformers import AutoTokenizer, AutoModelForCausalLM
from config import (
    SAFETY_MODEL_NAME, CACHE_DIR,
    RD_FAITH_PATH, RD_FAITH_LOGITS_PATH,
    RD_SAFETY_PATH, RD_SAFETY_LOGITS_PATH,
    CANDIDATE_N, SOFT_STRENGTH,
)
from candidates import select_candidates, load_rd_scores
from intervene import ExpertSteerer
from load_safety import load_advbench, FORCED_PREFIX, SAFETY_SYSTEM_PROMPT
from load_faith import load_faitheval_counterfactual

OUTPUT_PATH = os.path.join(_HERE, "..", "validation_result.json")
N = 200


# ---------------------------------------------------------------------------
# Input preparation — mirrors run_stage2.py tokenisation exactly
# ---------------------------------------------------------------------------

def _safety_neg_input_ids(model, tokenizer, prompts):
    """AdvBench prompts with forced harmful prefix — mirrors generate_forced (safety_safe task).
    Used to validate suppression of compliance-preferred experts: this is the condition
    where those experts are most active."""
    ids_list = []
    prefix_ids = tokenizer(
        FORCED_PREFIX, add_special_tokens=False, return_tensors="pt"
    )["input_ids"].to(model.device)
    for prompt in prompts:
        base_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)
        ids_list.append(torch.cat([base_ids, prefix_ids], dim=-1))
    return ids_list


def _safety_pos_input_ids(model, tokenizer, prompts):
    """AdvBench prompts with safety system prompt — mirrors generate_safe_sp (safety_unsafe task).
    Used to validate suppression of refusal-preferred experts: this is the condition
    where those experts are most active."""
    ids_list = []
    for prompt in prompts:
        ids_list.append(
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": SAFETY_SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                return_tensors="pt",
            ).to(model.device)
        )
    return ids_list


def _faith_input_ids(model, tokenizer, records):
    """FaithEval records with context — mirrors generate_faith."""
    ids_list = []
    for rec in records:
        header = f"Context:\n{rec['context']}\n\nQuestion: {rec['question']}"
        if rec["options"]:
            opts = "\n".join(f"{k}. {v}" for k, v in rec["options"].items())
            prompt = f"{header}\n\nOptions:\n{opts}\n\nAnswer with a single letter (A, B, C, or D)."
        else:
            prompt = (
                f"{header}\n\nAnswer based only on the context. "
                "If the context does not contain enough information to answer, say so."
            )
        ids_list.append(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                return_tensors="pt",
            ).to(model.device)
        )
    return ids_list


# ---------------------------------------------------------------------------
# Routing data collection
# ---------------------------------------------------------------------------

def collect_routing_stats(model, ids_list):
    """
    Run a list of pre-tokenised input_ids through the model (forward pass only).
    Observation hooks are registered here, AFTER any ExpertSteerer hooks the
    caller has already attached — so they see the post-intervention state.

    Returns:
        {layer_idx (int): {
            "top_experts_flat": [int, ...],   # all selected expert slots across all tokens
            "mean_logits":      [float, ...], # mean pre-softmax gate logit per expert
            "n_tokens":         int,
        }}
    """
    layer_data = {}
    obs_hooks  = []

    for layer_idx, layer in enumerate(model.model.layers):
        if not hasattr(layer, "mlp") or not hasattr(layer.mlp, "gate"):
            continue
        layer_data[layer_idx] = {"top_experts_flat": [], "logit_sum": None, "n_tokens": 0}

        def _make_hook(lidx):
            def _hook(module, inputs, outputs):
                # inputs[0]: hidden state fed to the gate.
                #   Baseline / hard : original h
                #   Soft            : h + delta_h  (soft pre-hook already applied)
                #
                # outputs[0]: topk expert index tensor (int).
                #   Baseline / soft : gate's natural top-k selection
                #   Hard            : post-replacement selection (suppressed experts removed)
                h    = inputs[0].detach().float()       # [bsz, seq, d_model]
                topk = outputs[0].detach().cpu()        # [bsz, seq, k] or [seq, k]

                bsz, seq_len, d = h.shape
                with torch.no_grad():
                    logits = F.linear(
                        h.reshape(-1, d), module.weight.float()
                    )                                   # [bsz*seq, n_experts]

                layer_data[lidx]["top_experts_flat"].extend(topk.reshape(-1).tolist())

                logit_mean = logits.mean(dim=0).cpu()
                if layer_data[lidx]["logit_sum"] is None:
                    layer_data[lidx]["logit_sum"] = logit_mean.clone()
                else:
                    layer_data[lidx]["logit_sum"] += logit_mean
                layer_data[lidx]["n_tokens"] += seq_len

            return _hook

        obs_hooks.append(layer.mlp.gate.register_forward_hook(_make_hook(layer_idx)))

    for ids in ids_list:
        with torch.no_grad():
            model(ids, use_cache=False)

    for h in obs_hooks:
        h.remove()

    result = {}
    n = len(ids_list)
    for lidx, d in layer_data.items():
        mean_logits = (d["logit_sum"] / n).tolist() if d["logit_sum"] is not None else []
        result[lidx] = {
            "top_experts_flat": d["top_experts_flat"],
            "mean_logits":      mean_logits,
            "n_tokens":         d["n_tokens"],
        }
    return result


def _routing_rate(stats, layer_idx, expert_idx):
    flat = stats[layer_idx]["top_experts_flat"]
    return flat.count(expert_idx) / len(flat) if flat else 0.0


def _mean_logit(stats, layer_idx, expert_idx):
    logits = stats[layer_idx]["mean_logits"]
    return logits[expert_idx] if expert_idx < len(logits) else float("nan")


# ---------------------------------------------------------------------------
# Per-axis validation
# ---------------------------------------------------------------------------

def validate_axis(model, ids_list, hard_candidates, soft_rd_scores, strength, label, direction, dataset):
    n_candidates = sum(len(v) for v in hard_candidates.values())
    print(f"\n{'='*60}")
    print(f"  Axis: {label}  |  direction: {direction}  |  candidates: {n_candidates}  |  n={len(ids_list)}  |  dataset: {dataset}")
    print(f"{'='*60}")

    print("  [1/3] Baseline (no steering)...")
    baseline_stats = collect_routing_stats(model, ids_list)

    print("  [2/3] Hard steering...")
    steerer = ExpertSteerer(model, hard_candidates, mode="hard", strength=strength)
    hard_stats = collect_routing_stats(model, ids_list)
    steerer.remove()

    print("  [3/3] Soft steering...")
    steerer = ExpertSteerer(model, soft_rd_scores, mode="soft", strength=strength)
    soft_stats = collect_routing_stats(model, ids_list)
    steerer.remove()

    layers_result = {}
    for layer_idx, expert_list in hard_candidates.items():
        experts_result = {}
        for ei in expert_list:
            b_rate  = _routing_rate(baseline_stats, layer_idx, ei)
            h_rate  = _routing_rate(hard_stats,     layer_idx, ei)
            s_rate  = _routing_rate(soft_stats,     layer_idx, ei)
            b_logit = _mean_logit(baseline_stats, layer_idx, ei)
            s_logit = _mean_logit(soft_stats,     layer_idx, ei)
            rd_val  = soft_rd_scores.get(layer_idx, {}).get(ei, float("nan"))
            exp_shift = strength * rd_val if rd_val == rd_val else float("nan")
            act_shift = s_logit - b_logit

            experts_result[str(ei)] = {
                "baseline_rate":        round(b_rate,    4),
                "hard_rate":            round(h_rate,    4),
                "soft_rate":            round(s_rate,    4),
                "baseline_mean_logit":  round(b_logit,   4),
                "soft_mean_logit":      round(s_logit,   4),
                "expected_logit_shift": round(exp_shift, 4) if exp_shift == exp_shift else None,
                "actual_logit_shift":   round(act_shift, 4),
                "hard_ok":              h_rate == 0.0,
                "soft_rate_reduced":    s_rate < b_rate or (s_rate == 0.0 and b_rate == 0.0),
            }

        layers_result[str(layer_idx)] = {"experts": experts_result}

    return layers_result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Loading model: {SAFETY_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(
        SAFETY_MODEL_NAME, cache_dir=CACHE_DIR, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        SAFETY_MODEL_NAME, cache_dir=CACHE_DIR,
        torch_dtype=torch.float16, device_map="auto", trust_remote_code=True,
    )
    model.eval()

    # Candidate selection — mirrors run_stage2.py exactly
    safety_hard = select_candidates(
        RD_SAFETY_PATH, RD_SAFETY_LOGITS_PATH, CANDIDATE_N, direction="negative"
    )
    safety_pos = select_candidates(
        RD_SAFETY_PATH, RD_SAFETY_LOGITS_PATH, CANDIDATE_N, direction="positive"
    )
    faith_hard = select_candidates(
        RD_FAITH_PATH, RD_FAITH_LOGITS_PATH, CANDIDATE_N, direction="negative"
    )
    _safety_full = load_rd_scores(RD_SAFETY_PATH, RD_SAFETY_LOGITS_PATH)
    _faith_full  = load_rd_scores(RD_FAITH_PATH,  RD_FAITH_LOGITS_PATH)
    safety_soft  = {l: _safety_full[l] for l in sorted(set(safety_hard) | set(safety_pos)) if l in _safety_full}
    faith_soft   = {l: _faith_full[l]  for l in faith_hard  if l in _faith_full}

    # Load actual experimental datasets
    print(f"Loading datasets (n={N})...")
    safety_prompts = load_advbench(n=N)
    faith_records  = load_faitheval_counterfactual(n=N)

    # Tokenise using the same format as run_stage2.py for each condition
    safety_neg_ids = _safety_neg_input_ids(model, tokenizer, safety_prompts)
    safety_pos_ids = _safety_pos_input_ids(model, tokenizer, safety_prompts)
    faith_ids      = _faith_input_ids(model, tokenizer, faith_records)

    safety_neg_result = validate_axis(
        model, safety_neg_ids, safety_hard, safety_soft, SOFT_STRENGTH,
        "safety", "negative", "AdvBench (forced prefix)",
    )
    safety_pos_result = validate_axis(
        model, safety_pos_ids, safety_pos, safety_soft, -SOFT_STRENGTH,
        "safety", "positive", "AdvBench (safety system prompt)",
    )
    faith_result = validate_axis(
        model, faith_ids, faith_hard, faith_soft, SOFT_STRENGTH,
        "faithfulness", "negative", "FaithEval-Counterfactual",
    )

    output = {
        "config": {
            "candidate_n":   CANDIDATE_N,
            "soft_strength": SOFT_STRENGTH,
            "n":             N,
        },
        "safety_negative": {
            "candidates": {str(k): v for k, v in safety_hard.items()},
            "layers":     safety_neg_result,
        },
        "safety_positive": {
            "candidates": {str(k): v for k, v in safety_pos.items()},
            "layers":     safety_pos_result,
        },
        "faithfulness": {
            "candidates": {str(k): v for k, v in faith_hard.items()},
            "layers":     faith_result,
        },
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nValidation result written to: {OUTPUT_PATH}")

    all_entries = [
        e
        for axis in (safety_neg_result, safety_pos_result, faith_result)
        for layer in axis.values()
        for e in layer["experts"].values()
    ]
    n_pass = sum(1 for e in all_entries if e["hard_ok"])
    n_total = len(all_entries)
    print(f"\nHard steering: {n_pass}/{n_total} experts pass (routing rate == 0.0)")
    if n_pass < n_total:
        print("  WARNING: some experts still fire under hard steering.")


if __name__ == "__main__":
    main()
