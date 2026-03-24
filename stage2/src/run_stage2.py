import sys
import os
import json
import re
import string
import argparse
import torch
from contextlib import nullcontext
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../stage1/src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../prep"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from transformers import AutoTokenizer, AutoModelForCausalLM

from config import (
    SAFETY_MODEL_NAME, CACHE_DIR,
    RD_FAITH_PATH, RD_FAITH_LOGITS_PATH,
    RD_SAFETY_PATH, RD_SAFETY_LOGITS_PATH,
    CANDIDATE_N, SOFT_STRENGTH, MAX_NEW_TOKENS, RESULTS_DIR,
)
from candidates import select_candidates, load_rd_scores
from intervene import ExpertSteerer
from classify import LlamaGuardClassifier
from load_safety import load_advbench, FORCED_PREFIX, SAFETY_SYSTEM_PROMPT
from load_faith import (
    load_faitheval_counterfactual,
    load_faitheval_unanswerable,
    load_mctest,
)

ALL_TASKS = ["safety_safe", "safety_unsafe", "faith_cf", "faith_un", "faith_mc", "fluency"]
ALL_CONDITIONS = ["baseline", "hard", "soft"]


def parse_args():
    parser = argparse.ArgumentParser(description="Run Stage 2 experiments.")
    parser.add_argument(
        "--tasks", nargs="+", choices=ALL_TASKS, default=ALL_TASKS,
        help="Which experiments to run (default: all).",
    )
    parser.add_argument(
        "--conditions", nargs="+", choices=ALL_CONDITIONS, default=ALL_CONDITIONS,
        help="Which conditions to run (default: baseline, hard, soft).",
    )
    parser.add_argument(
        "--n", type=int, default=100,
        help="Number of prompts/records per dataset (default: 100).",
    )
    parser.add_argument(
        "--candidate_n", type=int, default=None,
        help="Override CANDIDATE_N from config.",
    )
    parser.add_argument(
        "--soft_strength", type=float, default=None,
        help="Override SOFT_STRENGTH from config.",
    )
    parser.add_argument(
        "--results_dir", type=str, default=None,
        help="Directory to write per-condition JSON files (default: from config).",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print each prompt, generation, and label to stdout.",
    )
    parser.add_argument(
        "--skip_done", action="store_true",
        help="Skip conditions whose output file already exists with complete records.",
    )
    args = parser.parse_args()
    return (
        args.tasks, args.conditions, args.n,
        args.candidate_n, args.soft_strength,
        args.results_dir, args.verbose, args.skip_done,
    )


# Generation

def _generate(model, tokenizer, input_ids):
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=False,
        )
    return tokenizer.decode(
        output_ids[0][input_ids.shape[-1]:],
        skip_special_tokens=True,
    ).strip()


def generate_forced(model, tokenizer, prompt):
    messages = [{"role": "user", "content": prompt}]
    base_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt",
    ).to(model.device)
    prefix_ids = tokenizer(
        FORCED_PREFIX, add_special_tokens=False, return_tensors="pt",
    )["input_ids"].to(model.device)
    input_ids = torch.cat([base_ids, prefix_ids], dim=-1)
    return FORCED_PREFIX + " " + _generate(model, tokenizer, input_ids)


def generate_safe_sp(model, tokenizer, prompt):
    chat = [
        {"role": "system", "content": SAFETY_SYSTEM_PROMPT},
        {"role": "user",   "content": prompt},
    ]
    input_ids = tokenizer.apply_chat_template(chat, return_tensors="pt").to(model.device)
    return _generate(model, tokenizer, input_ids)


def generate_plain(model, tokenizer, prompt):
    """Plain generation with no system prompt — used for fluency check."""
    messages = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt",
    ).to(model.device)
    return _generate(model, tokenizer, input_ids)


def generate_faith(model, tokenizer, record):
    messages = [{"role": "user", "content": _build_faith_prompt(record)}]
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt",
    ).to(model.device)
    return _generate(model, tokenizer, input_ids)


def _build_faith_prompt(record):
    header = f"Context:\n{record['context']}\n\nQuestion: {record['question']}"
    if record["options"]:
        opts = "\n".join(f"{k}. {v}" for k, v in record["options"].items())
        return f"{header}\n\nOptions:\n{opts}\n\nAnswer with a single letter (A, B, C, or D)."
    return (
        f"{header}\n\n"
        "Answer based only on the context. "
        "If the context does not contain enough information to answer, say so."
    )


def _extract_mcq_letter(text):
    m = re.search(r"\b([A-D])\b", text.upper())
    return m.group(1) if m else ""


def _normalise(text):
    text = str(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def _faith_correct(pred, gold, options):
    """pred is letter-extracted for MCQ, raw string for open-ended."""
    if options:
        return pred.upper() == str(gold).upper()
    elif isinstance(gold, list):
        return any(_normalise(pred) == _normalise(g) for g in gold)
    else:
        return _normalise(pred) == _normalise(str(gold))


# Per-condition file I/O

def _condition_path(results_dir, task, condition):
    return os.path.join(results_dir, task, f"{condition}.json")


def _save_condition(results_dir, task, condition, records, metric_key, metric_val, n_target):
    os.makedirs(os.path.join(results_dir, task), exist_ok=True)
    is_steered_condition = (condition != "baseline")
    data = {
        "task": task,
        "condition": condition,
        "is_steered_condition": is_steered_condition,
        "n_target": n_target,
        "n_complete": len(records),
        metric_key: metric_val,
        "steered_indices": [r["idx"] for r in records if r.get("steered")],
        "records": records,
    }

    if task.startswith("safety_"):
        # Safety files include an indexed view for fast scanning of prompt/response/safety/category tuples.
        data["indexed_prompt_response_safe_category"] = [
            {
                "idx": r["idx"],
                "prompt": r["prompt"],
                "response": r["response"],
                "safe": r.get("safe"),
                "category": r.get("category"),
            }
            for r in records
        ]

    path = _condition_path(results_dir, task, condition)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _is_complete(results_dir, task, condition, n_target):
    path = _condition_path(results_dir, task, condition)
    if not os.path.exists(path):
        return False
    try:
        with open(path) as f:
            d = json.load(f)
        return len(d.get("records", [])) >= n_target
    except (json.JSONDecodeError, KeyError):
        return False


def _annotate_mismatches(results_dir, task, cond, mismatch_key):
    """
    Compare steered condition file against baseline and annotate each record with:
      baseline_<key>: the baseline value for that idx
      mismatch:       True where the steered value differs from baseline
    Also writes top-level n_mismatches and mismatch_indices to the steered file.
    No-ops if baseline file doesn't exist yet.
    """
    baseline_path = _condition_path(results_dir, task, "baseline")
    steered_path  = _condition_path(results_dir, task, cond)
    if not os.path.exists(baseline_path) or not os.path.exists(steered_path):
        return
    with open(baseline_path) as f:
        baseline_data = json.load(f)
    baseline_by_idx = {r["idx"]: r.get(mismatch_key) for r in baseline_data["records"]}
    with open(steered_path) as f:
        data = json.load(f)
    mismatch_idxs = []
    mismatch_records = []
    for r in data["records"]:
        bval = baseline_by_idx.get(r["idx"])
        r[f"baseline_{mismatch_key}"] = bval
        r["mismatch"] = (r.get(mismatch_key) != bval) if bval is not None else None
        if r.get("mismatch"):
            mismatch_idxs.append(r["idx"])
            mismatch_records.append({
                "idx": r["idx"],
                "prompt": r.get("prompt"),
                "response": r.get("response"),
                mismatch_key: r.get(mismatch_key),
                f"baseline_{mismatch_key}": bval,
                "category": r.get("category"),
                "steered": r.get("steered"),
            })
    data["n_mismatches"] = len(mismatch_idxs)
    data["mismatch_indices"] = mismatch_idxs
    data["mismatches"] = mismatch_records
    with open(steered_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  [{cond}] mismatches vs baseline: {len(mismatch_idxs)}/{len(data['records'])}")


# Aggregate metrics from record lists

def _safe_rate_from_records(records):
    if not records:
        return 0.0
    return sum(1 for r in records if r.get("safe")) / len(records)


def _accuracy_from_records(records):
    if not records:
        return 0.0
    return sum(1 for r in records if r.get("correct")) / len(records)


def _mean_length_from_records(records):
    if not records:
        return 0.0
    return sum(len(r["response"].split()) for r in records) / len(records)


# Batch runners

def run_safety_batch(
    model, tokenizer, classifier, prompts, gen_fn,
    candidates, mode, strength, steered, verbose=False,
):
    ctx = ExpertSteerer(model, candidates, mode=mode, strength=strength) if candidates else nullcontext()
    records = []
    with ctx:
        for idx, p in enumerate(tqdm(prompts, desc="prompts", leave=False)):
            response = gen_fn(model, tokenizer, p)
            label = classifier.classify(p, response)
            records.append({
                "idx": idx,
                "prompt": p,
                "response": response,
                "steered": steered,
                **label,
            })
            if verbose:
                print(f"\n--- [{idx}] PROMPT ---\n{p}")
                print(f"--- RESPONSE ---\n{response}")
                print(f"--- LABEL ---\n{label}")
    return records


def run_faith_batch(
    model, tokenizer, records_in,
    candidates, mode, strength, steered,
):
    ctx = ExpertSteerer(model, candidates, mode=mode, strength=strength) if candidates else nullcontext()
    records_out = []
    with ctx:
        for idx, rec in enumerate(tqdm(records_in, desc="records", leave=False)):
            raw = generate_faith(model, tokenizer, rec)
            pred = _extract_mcq_letter(raw) if rec["options"] else raw
            correct = _faith_correct(pred, rec["gold"], rec["options"])
            records_out.append({
                "idx": idx,
                "prompt": _build_faith_prompt(rec),
                "response": raw,
                "prediction": pred,
                "gold": rec["gold"],
                "correct": correct,
                "steered": steered,
            })
    return records_out


def run_fluency_batch(
    model, tokenizer, prompts,
    candidates, mode, strength, steered, verbose=False,
):
    """
    Generate plain responses on innocuous HH-RLHF Harmless prompts.
    Used to check that safety steering does not degrade general fluency.
    """
    ctx = ExpertSteerer(model, candidates, mode=mode, strength=strength) if candidates else nullcontext()
    records = []
    with ctx:
        for idx, p in enumerate(tqdm(prompts, desc="prompts", leave=False)):
            response = generate_plain(model, tokenizer, p)
            records.append({
                "idx": idx,
                "prompt": p,
                "response": response,
                "steered": steered,
            })
            if verbose:
                print(f"\n--- [{idx}] PROMPT ---\n{p}")
                print(f"--- RESPONSE ---\n{response}")
    return records


# Condition builder

def _build_conditions(selected_conditions, safety_neg, safety_pos, faith_neg,
                       safety_scores, faith_scores, strength):
    """
    Return condition lists filtered to only selected conditions.
    Each entry: (name, candidates, mode, effective_strength).
    """
    all_safe = [
        ("baseline", None,          "hard", strength),
        ("hard",     safety_neg,    "hard", strength),
        ("soft",     safety_scores, "soft", strength),
    ]
    all_unsafe = [
        ("baseline", None,          "hard", strength),
        ("hard",     safety_pos,    "hard", strength),
        ("soft",     safety_scores, "soft", -strength),
    ]
    all_faith = [
        ("baseline", None,         "hard", strength),
        ("hard",     faith_neg,    "hard", strength),
        ("soft",     faith_scores, "soft", strength),
    ]
    filt = lambda lst: [(n, c, m, s) for (n, c, m, s) in lst if n in selected_conditions]
    return filt(all_safe), filt(all_unsafe), filt(all_faith)


# Main

def main():
    tasks, conditions, n, candidate_n_override, soft_strength_override, results_dir_override, verbose, skip_done = parse_args()
    candidate_n  = candidate_n_override   if candidate_n_override   is not None else CANDIDATE_N
    soft_strength = soft_strength_override if soft_strength_override is not None else SOFT_STRENGTH
    results_dir  = results_dir_override   if results_dir_override   is not None else RESULTS_DIR

    os.makedirs(results_dir, exist_ok=True)
    print(f"Results dir: {results_dir}")

    print(f"Loading {SAFETY_MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(
        SAFETY_MODEL_NAME, cache_dir=CACHE_DIR, trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        SAFETY_MODEL_NAME, cache_dir=CACHE_DIR,
        torch_dtype=torch.float16, device_map="auto", trust_remote_code=True,
    )
    model.eval()

    needs_classifier = any(t in tasks for t in ["safety_safe", "safety_unsafe"])
    classifier = LlamaGuardClassifier(cache_dir=CACHE_DIR) if needs_classifier else None

    safety_neg   = select_candidates(RD_SAFETY_PATH, RD_SAFETY_LOGITS_PATH, candidate_n, direction="negative")
    safety_pos   = select_candidates(RD_SAFETY_PATH, RD_SAFETY_LOGITS_PATH, candidate_n, direction="positive")
    faith_neg    = select_candidates(RD_FAITH_PATH,  RD_FAITH_LOGITS_PATH,  candidate_n, direction="negative")
    _safety_scores_full = load_rd_scores(RD_SAFETY_PATH, RD_SAFETY_LOGITS_PATH)
    _faith_scores_full  = load_rd_scores(RD_FAITH_PATH,  RD_FAITH_LOGITS_PATH)

    safety_scores = {l: _safety_scores_full[l] for l in sorted(set(safety_neg) | set(safety_pos)) if l in _safety_scores_full}
    faith_scores  = {l: _faith_scores_full[l]  for l in faith_neg if l in _faith_scores_full}

    safe_conds, unsafe_conds, faith_conds = _build_conditions(
        conditions, safety_neg, safety_pos, faith_neg,
        safety_scores, faith_scores, soft_strength,
    )
    # Fluency uses same candidates as safe steering (safety-neg experts)
    fluency_conds = list(safe_conds)

    print("Loading datasets...")
    safety_prompts  = load_advbench(n=n)    if any(t in tasks for t in ["safety_safe", "safety_unsafe"]) else []
    fluency_prompts = []  # fluency task requires HH-RLHF loader — not yet implemented
    cf_records      = load_faitheval_counterfactual(n=n) if "faith_cf" in tasks else []
    un_records      = load_faitheval_unanswerable(n=n)   if "faith_un" in tasks else []
    mc_records      = load_mctest(n=n)                   if "faith_mc" in tasks else []

    # ------------------------------------------------------------------
    # Safety: safe steering — suppress unsafe experts on forced-prefix prompts
    # ------------------------------------------------------------------
    if "safety_safe" in tasks:
        print("\nSafety: safe steering (forced prefix + suppress unsafe experts)...")
        for cond_name, cands, mode, strength in safe_conds:
            if skip_done and _is_complete(results_dir, "safety_safe", cond_name, n):
                print(f"  [{cond_name}] skipping (complete).")
                continue
            print(f"  [{cond_name}] running...")
            records = run_safety_batch(
                model, tokenizer, classifier, safety_prompts,
                generate_forced, cands, mode, strength,
                steered=(cond_name != "baseline"), verbose=verbose,
            )
            sr = _safe_rate_from_records(records)
            _save_condition(results_dir, "safety_safe", cond_name, records, "safe_rate", sr, n)
            print(f"  [{cond_name}] safe_rate: {sr:.3f}")
            if cond_name != "baseline":
                _annotate_mismatches(results_dir, "safety_safe", cond_name, "safe")

    # ------------------------------------------------------------------
    # Safety: unsafe steering — suppress safe experts on safety-SP prompts
    # ------------------------------------------------------------------
    if "safety_unsafe" in tasks:
        print("\nSafety: unsafe steering (safety system prompt + suppress safe experts)...")
        for cond_name, cands, mode, strength in unsafe_conds:
            if skip_done and _is_complete(results_dir, "safety_unsafe", cond_name, n):
                print(f"  [{cond_name}] skipping (complete).")
                continue
            print(f"  [{cond_name}] running...")
            records = run_safety_batch(
                model, tokenizer, classifier, safety_prompts,
                generate_safe_sp, cands, mode, strength,
                steered=(cond_name != "baseline"), verbose=verbose,
            )
            sr = _safe_rate_from_records(records)
            _save_condition(results_dir, "safety_unsafe", cond_name, records, "safe_rate", sr, n)
            print(f"  [{cond_name}] safe_rate: {sr:.3f}")
            if cond_name != "baseline":
                _annotate_mismatches(results_dir, "safety_unsafe", cond_name, "safe")

    # ------------------------------------------------------------------
    # Faithfulness tasks
    # ------------------------------------------------------------------
    faith_task_map = [
        ("faith_cf", cf_records),
        ("faith_un", un_records),
        ("faith_mc", mc_records),
    ]
    for task_key, dataset_records in faith_task_map:
        if task_key not in tasks:
            continue
        print(f"\nFaithfulness: {task_key}...")
        for cond_name, cands, mode, strength in faith_conds:
            if skip_done and _is_complete(results_dir, task_key, cond_name, n):
                print(f"  [{cond_name}] skipping (complete).")
                continue
            print(f"  [{cond_name}] running...")
            records = run_faith_batch(
                model, tokenizer, dataset_records, cands, mode, strength,
                steered=(cond_name != "baseline"),
            )
            acc = _accuracy_from_records(records)
            _save_condition(results_dir, task_key, cond_name, records, "accuracy", acc, n)
            print(f"  [{cond_name}] accuracy: {acc:.3f}")
            if cond_name != "baseline":
                _annotate_mismatches(results_dir, task_key, cond_name, "correct")

    # ------------------------------------------------------------------
    # Fluency check (HH-RLHF Harmless — innocuous prompts, safety steering)
    # ------------------------------------------------------------------
    if "fluency" in tasks:
        print("\nFluency check (HH-RLHF Harmless, safety-neg candidates)...")
        for cond_name, cands, mode, strength in fluency_conds:
            if skip_done and _is_complete(results_dir, "fluency", cond_name, n):
                print(f"  [{cond_name}] skipping (complete).")
                continue
            print(f"  [{cond_name}] running...")
            records = run_fluency_batch(
                model, tokenizer, fluency_prompts, cands, mode, strength,
                steered=(cond_name != "baseline"), verbose=verbose,
            )
            mean_len = _mean_length_from_records(records)
            _save_condition(results_dir, "fluency", cond_name, records,
                            "mean_response_length", mean_len, n)
            print(f"  [{cond_name}] mean response length: {mean_len:.1f} words")

    print(f"\nDone. Results written to {results_dir}/")


if __name__ == "__main__":
    main()
