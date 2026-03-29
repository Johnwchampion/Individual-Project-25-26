#!/usr/bin/env python3
"""
Mechanistic validation of ExpertSteerer hooks.

Runs 200 prompts from the actual experimental datasets (AdvBench forced prefix
for safety, FaithEval-Counterfactual for faithfulness) under baseline, hard,
and soft conditions. For each targeted (layer, expert) pair, checks that
hard steering drives routing rate to zero and soft steering reduces it relative
to baseline. Writes results to stage2/validation_result.json.
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


# Input preparation — mirrors run_stage2.py tokenisation exactly

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


# Routing data collection

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


# Per-axis validation

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


# Token-range validation (faithfulness question-span steering)

def _find_subsequence(seq, subseq):
    for i in range(len(seq) - len(subseq) + 1):
        if seq[i:i + len(subseq)] == subseq:
            return i
    return None


def _faith_ids_with_spans(model, tokenizer, records):
    """
    Returns list of (input_ids, q_start, q_end) for each faith record.
    q_start/q_end are the token indices of the question text within the
    full chat-templated input — the same span used during Stage 1 RD measurement.
    Records where the question span cannot be located are skipped.
    """
    result = []
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
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)
        full_ids = ids[0].tolist()
        q_ids = tokenizer(" " + rec["question"], add_special_tokens=False)["input_ids"]
        start = _find_subsequence(full_ids, q_ids)
        if start is None:
            continue
        result.append((ids, start, start + len(q_ids)))
    return result


def collect_routing_stats_split(model, ids_with_spans):
    """
    Like collect_routing_stats but separately accumulates routing stats for
    question-span tokens [q_start:q_end] and all other (context) tokens.

    ids_with_spans: list of (input_ids, q_start, q_end)

    Returns:
        {layer_idx: {
            "q_experts": [int, ...],   # expert slots for question-span tokens
            "c_experts": [int, ...],   # expert slots for context tokens
            "q_logits":  [float, ...], # mean gate logit per expert (question-span)
            "c_logits":  [float, ...], # mean gate logit per expert (context)
            "n_q_tokens": int,
            "n_c_tokens": int,
        }}
    """
    layer_data = {}
    for layer_idx, layer in enumerate(model.model.layers):
        if not hasattr(layer, "mlp") or not hasattr(layer.mlp, "gate"):
            continue
        layer_data[layer_idx] = {
            "q_experts": [], "c_experts": [],
            "logit_sum_q": None, "logit_sum_c": None,
            "n_q_tokens": 0, "n_c_tokens": 0,
        }

    current_span = [None]
    obs_hooks = []

    for layer_idx, layer in enumerate(model.model.layers):
        if not hasattr(layer, "mlp") or not hasattr(layer.mlp, "gate"):
            continue

        def _make_hook(lidx):
            def _hook(module, inputs, outputs):
                h    = inputs[0].detach().float()   # [1, seq_len, d_model]
                topk = outputs[0].detach().cpu()    # [seq_len, k] or [1, seq_len, k]
                seq_len = h.shape[1]
                with torch.no_grad():
                    logits = F.linear(
                        h.reshape(-1, h.shape[-1]), module.weight.float()
                    )  # [seq_len, n_experts]

                topk_by_token = topk.reshape(seq_len, -1)  # [seq_len, k]
                q_start, q_end = current_span[0]
                q_end = min(q_end, seq_len)

                # Question-span
                if q_start < q_end:
                    q_topk = topk_by_token[q_start:q_end]
                    layer_data[lidx]["q_experts"].extend(q_topk.reshape(-1).tolist())
                    layer_data[lidx]["n_q_tokens"] += q_end - q_start
                    q_mean = logits[q_start:q_end].mean(dim=0).cpu()
                    if layer_data[lidx]["logit_sum_q"] is None:
                        layer_data[lidx]["logit_sum_q"] = q_mean.clone()
                    else:
                        layer_data[lidx]["logit_sum_q"] += q_mean

                # Context (everything outside question span)
                c_idx = list(range(0, q_start)) + list(range(q_end, seq_len))
                if c_idx:
                    c_topk = topk_by_token[c_idx]
                    layer_data[lidx]["c_experts"].extend(c_topk.reshape(-1).tolist())
                    layer_data[lidx]["n_c_tokens"] += len(c_idx)
                    c_mean = logits[c_idx].mean(dim=0).cpu()
                    if layer_data[lidx]["logit_sum_c"] is None:
                        layer_data[lidx]["logit_sum_c"] = c_mean.clone()
                    else:
                        layer_data[lidx]["logit_sum_c"] += c_mean

            return _hook

        obs_hooks.append(layer.mlp.gate.register_forward_hook(_make_hook(layer_idx)))

    for ids, q_start, q_end in ids_with_spans:
        current_span[0] = (q_start, q_end)
        with torch.no_grad():
            model(ids, use_cache=False)

    for h in obs_hooks:
        h.remove()

    n = len(ids_with_spans)
    result = {}
    for lidx, d in layer_data.items():
        result[lidx] = {
            "q_experts":  d["q_experts"],
            "c_experts":  d["c_experts"],
            "q_logits":   (d["logit_sum_q"] / n).tolist() if d["logit_sum_q"] is not None else [],
            "c_logits":   (d["logit_sum_c"] / n).tolist() if d["logit_sum_c"] is not None else [],
            "n_q_tokens": d["n_q_tokens"],
            "n_c_tokens": d["n_c_tokens"],
        }
    return result


def _split_rate(stats, layer_idx, expert_idx, span):
    """Rate for question-span ('q') or context ('c') tokens."""
    flat = stats[layer_idx][f"{span}_experts"]
    return flat.count(expert_idx) / len(flat) if flat else 0.0


def _split_logit(stats, layer_idx, expert_idx, span):
    logits = stats[layer_idx][f"{span}_logits"]
    return logits[expert_idx] if expert_idx < len(logits) else float("nan")


def validate_faith_token_range(model, tokenizer, faith_records, hard_candidates, soft_rd_scores, strength):
    """
    Validates that token-range-restricted steering:
      - fires on question-span tokens (hard rate == 0.0, soft rate reduced)
      - does NOT fire on context tokens (rates unchanged from baseline)
    """
    print("\n" + "=" * 60)
    print("  Token-range validation: faithfulness question-span steering")
    print("=" * 60)

    ids_with_spans = _faith_ids_with_spans(model, tokenizer, faith_records)
    print(f"  Records with located question span: {len(ids_with_spans)}/{len(faith_records)}")

    print("  [1/3] Baseline (no steering)...")
    baseline = collect_routing_stats_split(model, ids_with_spans)

    print("  [2/3] Hard steering (question-span only)...")
    hard_cands_with_range = {}
    for ids, q_start, q_end in ids_with_spans:
        # token_range varies per record — run one forward pass at a time
        break  # we'll handle per-record below
    # Run hard steering with per-record token_range
    hard_layer_data = {
        lidx: {"q_experts": [], "c_experts": [],
               "logit_sum_q": None, "logit_sum_c": None,
               "n_q_tokens": 0, "n_c_tokens": 0}
        for lidx in baseline
    }
    current_span = [None]
    obs_hooks = []
    for layer_idx, layer in enumerate(model.model.layers):
        if not hasattr(layer, "mlp") or not hasattr(layer.mlp, "gate"):
            continue
        def _make_hook(lidx):
            def _hook(module, inputs, outputs):
                h    = inputs[0].detach().float()
                topk = outputs[0].detach().cpu()
                seq_len = h.shape[1]
                with torch.no_grad():
                    logits = F.linear(h.reshape(-1, h.shape[-1]), module.weight.float())
                topk_by_token = topk.reshape(seq_len, -1)
                q_start, q_end = current_span[0]
                q_end = min(q_end, seq_len)
                if q_start < q_end:
                    hard_layer_data[lidx]["q_experts"].extend(topk_by_token[q_start:q_end].reshape(-1).tolist())
                    hard_layer_data[lidx]["n_q_tokens"] += q_end - q_start
                c_idx = list(range(0, q_start)) + list(range(q_end, seq_len))
                if c_idx:
                    hard_layer_data[lidx]["c_experts"].extend(topk_by_token[c_idx].reshape(-1).tolist())
                    hard_layer_data[lidx]["n_c_tokens"] += len(c_idx)
            return _hook
        obs_hooks.append(layer.mlp.gate.register_forward_hook(_make_hook(layer_idx)))

    for ids, q_start, q_end in ids_with_spans:
        current_span[0] = (q_start, q_end)
        steerer = ExpertSteerer(model, hard_candidates, mode="hard", strength=strength,
                                token_range=(q_start, q_end))
        with torch.no_grad():
            model(ids, use_cache=False)
        steerer.remove()

    for h in obs_hooks:
        h.remove()
    n = len(ids_with_spans)
    hard = {lidx: {
        "q_experts": d["q_experts"], "c_experts": d["c_experts"],
        "q_logits": [], "c_logits": [],
        "n_q_tokens": d["n_q_tokens"], "n_c_tokens": d["n_c_tokens"],
    } for lidx, d in hard_layer_data.items()}

    print("  [3/3] Soft steering (question-span only)...")
    soft_layer_data = {
        lidx: {"q_experts": [], "c_experts": [],
               "logit_sum_q": None, "logit_sum_c": None,
               "n_q_tokens": 0, "n_c_tokens": 0}
        for lidx in baseline
    }
    current_span = [None]
    obs_hooks = []
    for layer_idx, layer in enumerate(model.model.layers):
        if not hasattr(layer, "mlp") or not hasattr(layer.mlp, "gate"):
            continue
        def _make_hook(lidx):
            def _hook(module, inputs, outputs):
                h    = inputs[0].detach().float()
                topk = outputs[0].detach().cpu()
                seq_len = h.shape[1]
                with torch.no_grad():
                    logits = F.linear(h.reshape(-1, h.shape[-1]), module.weight.float())
                topk_by_token = topk.reshape(seq_len, -1)
                q_start, q_end = current_span[0]
                q_end = min(q_end, seq_len)
                if q_start < q_end:
                    soft_layer_data[lidx]["q_experts"].extend(topk_by_token[q_start:q_end].reshape(-1).tolist())
                    soft_layer_data[lidx]["n_q_tokens"] += q_end - q_start
                    q_mean = logits[q_start:q_end].mean(dim=0).cpu()
                    if soft_layer_data[lidx]["logit_sum_q"] is None:
                        soft_layer_data[lidx]["logit_sum_q"] = q_mean.clone()
                    else:
                        soft_layer_data[lidx]["logit_sum_q"] += q_mean
                c_idx = list(range(0, q_start)) + list(range(q_end, seq_len))
                if c_idx:
                    soft_layer_data[lidx]["c_experts"].extend(topk_by_token[c_idx].reshape(-1).tolist())
                    soft_layer_data[lidx]["n_c_tokens"] += len(c_idx)
                    c_mean = logits[c_idx].mean(dim=0).cpu()
                    if soft_layer_data[lidx]["logit_sum_c"] is None:
                        soft_layer_data[lidx]["logit_sum_c"] = c_mean.clone()
                    else:
                        soft_layer_data[lidx]["logit_sum_c"] += c_mean
            return _hook
        obs_hooks.append(layer.mlp.gate.register_forward_hook(_make_hook(layer_idx)))

    for ids, q_start, q_end in ids_with_spans:
        current_span[0] = (q_start, q_end)
        steerer = ExpertSteerer(model, soft_rd_scores, mode="soft", strength=strength,
                                token_range=(q_start, q_end))
        with torch.no_grad():
            model(ids, use_cache=False)
        steerer.remove()

    for h in obs_hooks:
        h.remove()
    soft = {lidx: {
        "q_experts": d["q_experts"], "c_experts": d["c_experts"],
        "q_logits": (d["logit_sum_q"] / n).tolist() if d["logit_sum_q"] is not None else [],
        "c_logits": (d["logit_sum_c"] / n).tolist() if d["logit_sum_c"] is not None else [],
        "n_q_tokens": d["n_q_tokens"], "n_c_tokens": d["n_c_tokens"],
    } for lidx, d in soft_layer_data.items()}

    # Build results per candidate expert
    layers_result = {}
    for layer_idx, expert_list in hard_candidates.items():
        if layer_idx not in baseline:
            continue
        experts_result = {}
        for ei in expert_list:
            bq = _split_rate(baseline, layer_idx, ei, "q")
            bc = _split_rate(baseline, layer_idx, ei, "c")
            hq = _split_rate(hard,     layer_idx, ei, "q")
            hc = _split_rate(hard,     layer_idx, ei, "c")
            sq = _split_rate(soft,     layer_idx, ei, "q")
            sc = _split_rate(soft,     layer_idx, ei, "c")
            b_logit = _split_logit(baseline, layer_idx, ei, "q")
            s_logit = _split_logit(soft,     layer_idx, ei, "q")
            rd_val  = soft_rd_scores.get(layer_idx, {}).get(ei, float("nan"))
            exp_shift = strength * rd_val if rd_val == rd_val else float("nan")
            experts_result[str(ei)] = {
                "baseline_q_rate":  round(bq, 4),
                "baseline_c_rate":  round(bc, 4),
                "hard_q_rate":      round(hq, 4),
                "hard_c_rate":      round(hc, 4),
                "soft_q_rate":      round(sq, 4),
                "soft_c_rate":      round(sc, 4),
                "hard_q_ok":        hq == 0.0,
                "hard_c_ok":        abs(hc - bc) < 0.005,
                "soft_q_reduced":   sq < bq or (sq == 0.0 and bq == 0.0),
                "soft_c_ok":        abs(sc - bc) < 0.005,
                "expected_logit_shift": round(exp_shift, 4) if exp_shift == exp_shift else None,
                "actual_logit_shift":   round(s_logit - b_logit, 4),
            }
        layers_result[str(layer_idx)] = {"experts": experts_result}

    all_e = [e for layer in layers_result.values() for e in layer["experts"].values()]
    n_hq  = sum(1 for e in all_e if e["hard_q_ok"])
    n_hc  = sum(1 for e in all_e if e["hard_c_ok"])
    n_sq  = sum(1 for e in all_e if e["soft_q_reduced"])
    n_sc  = sum(1 for e in all_e if e["soft_c_ok"])
    total = len(all_e)
    print(f"  Hard Q (== 0.0):      {n_hq}/{total}")
    print(f"  Hard C (unchanged):   {n_hc}/{total}")
    print(f"  Soft Q (reduced):     {n_sq}/{total}")
    print(f"  Soft C (unchanged):   {n_sc}/{total}")
    return layers_result


# Main

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

    faith_token_range_result = validate_faith_token_range(
        model, tokenizer, faith_records, faith_hard, faith_soft, SOFT_STRENGTH,
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
        "faithfulness_token_range": {
            "description": (
                "Token-range-restricted steering: hooks fire only on question-span "
                "tokens [q_start:q_end], matching the Stage 1 RD measurement site. "
                "hard_q_ok=True means the targeted expert was fully suppressed on "
                "question tokens. hard_c_ok=True means context tokens were unaffected."
            ),
            "candidates": {str(k): v for k, v in faith_hard.items()},
            "layers":     faith_token_range_result,
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

    tr_entries = [
        e
        for layer in faith_token_range_result.values()
        for e in layer["experts"].values()
    ]
    n_hq = sum(1 for e in tr_entries if e["hard_q_ok"])
    n_hc = sum(1 for e in tr_entries if e["hard_c_ok"])
    n_sq = sum(1 for e in tr_entries if e["soft_q_reduced"])
    n_sc = sum(1 for e in tr_entries if e["soft_c_ok"])
    n_tr = len(tr_entries)
    print(f"\nToken-range validation:")
    print(f"  Hard  Q suppressed (== 0.0):  {n_hq}/{n_tr}")
    print(f"  Hard  C unchanged  (< 0.005): {n_hc}/{n_tr}")
    print(f"  Soft  Q reduced:              {n_sq}/{n_tr}")
    print(f"  Soft  C unchanged  (< 0.005): {n_sc}/{n_tr}")


if __name__ == "__main__":
    main()
