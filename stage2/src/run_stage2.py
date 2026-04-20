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
    FAITH_MODEL_NAME, SAFETY_MODEL_NAME, CACHE_DIR,
    RD_FAITH_PATH, RD_FAITH_LOGITS_PATH,
    RD_SAFETY_PATH, RD_SAFETY_LOGITS_PATH,
    CANDIDATE_N, SOFT_STRENGTH, MAX_NEW_TOKENS, RESULTS_DIR,
)
from candidates import select_candidates, select_random_candidates, load_rd_scores
from intervene import ExpertSteerer
from classify import LlamaGuardClassifier
from load_safety import load_advbench, FORCED_PREFIX, SAFETY_SYSTEM_PROMPT
from load_faith import (
    load_faitheval_counterfactual,
    load_race,
)

ALL_TASKS = ["safety_safe", "safety_unsafe", "faith_cf", "faith_unfaith", "faith_race", "faith_unfaith_race", "fluency"]
ALL_CONDITIONS = ["baseline", "hard", "soft", "random"]


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
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for stochastic sampling. If set, do_sample=True is used and "
             "results are written to <results_dir>/seed_<seed>/.",
    )
    parser.add_argument(
        "--random_seed", type=int, default=42,
        help="Seed for random candidate selection (select_random_candidates). "
             "When non-default, results are written to <results_dir>/random_seed_<N>/.",
    )
    args = parser.parse_args()
    return (
        args.tasks, args.conditions, args.n,
        args.candidate_n, args.soft_strength,
        args.results_dir, args.verbose, args.skip_done, args.seed,
        args.random_seed,
    )


# Generation

def _generate(model, tokenizer, input_ids, max_new_tokens=MAX_NEW_TOKENS, do_sample=False):
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=False,
        )
    return tokenizer.decode(
        output_ids[0][input_ids.shape[-1]:],
        skip_special_tokens=True,
    ).strip()


def generate_forced(model, tokenizer, prompt, do_sample=False):
    messages = [{"role": "user", "content": prompt}]
    base_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt",
    ).to(model.device)
    prefix_ids = tokenizer(
        FORCED_PREFIX, add_special_tokens=False, return_tensors="pt",
    )["input_ids"].to(model.device)
    input_ids = torch.cat([base_ids, prefix_ids], dim=-1)
    return FORCED_PREFIX + " " + _generate(model, tokenizer, input_ids, do_sample=do_sample)


def generate_safe_sp(model, tokenizer, prompt, do_sample=False):
    chat = [
        {"role": "system", "content": SAFETY_SYSTEM_PROMPT},
        {"role": "user",   "content": prompt},
    ]
    input_ids = tokenizer.apply_chat_template(chat, return_tensors="pt").to(model.device)
    return _generate(model, tokenizer, input_ids, do_sample=do_sample)


def generate_plain(model, tokenizer, prompt, do_sample=False):
    """Generate plain fluency-check text."""
    messages = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt",
    ).to(model.device)
    return _generate(model, tokenizer, input_ids, do_sample=do_sample)


def generate_faith(model, tokenizer, record, do_sample=False):
    is_mcq = bool(record["options"])
    messages = [{"role": "user", "content": _build_faith_prompt(record)}]
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt",
    ).to(model.device)
    if is_mcq:
        # Prime the model with "Answer:" so first token is the letter
        answer_prefix_ids = tokenizer(
            "Answer:", add_special_tokens=False, return_tensors="pt",
        )["input_ids"].to(model.device)
        input_ids = torch.cat([input_ids, answer_prefix_ids], dim=-1)
        return "Answer:" + _generate(model, tokenizer, input_ids, max_new_tokens=5, do_sample=do_sample)
    return _generate(model, tokenizer, input_ids, do_sample=do_sample)


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


def _find_subsequence(seq, subseq):
    for i in range(len(seq) - len(subseq) + 1):
        if seq[i:i + len(subseq)] == subseq:
            return i
    return None


def _get_question_token_range(tokenizer, record, full_ids):
    """Return the question token span within full_ids, or None."""
    question_ids = tokenizer(" " + record["question"], add_special_tokens=False)["input_ids"]
    start = _find_subsequence(full_ids, question_ids)
    if start is None:
        return None
    return (start, start + len(question_ids))


def _extract_mcq_letter(text):
    m = re.search(r"\b([A-D])\b", text.upper())
    return m.group(1) if m else ""


def _normalise(text):
    text = str(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def _faith_correct(pred, gold, options):
    """Compare MCQ letters or open-ended strings."""
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
    """Annotate steered records against baseline."""
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
    candidates, mode, strength, steered, do_sample=False, verbose=False,
    checkpoint_fn=None,
):
    ctx = ExpertSteerer(model, candidates, mode=mode, strength=strength) if candidates else nullcontext()
    records = []
    with ctx:
        for idx, p in enumerate(tqdm(prompts, desc="prompts", leave=False)):
            response = gen_fn(model, tokenizer, p, do_sample=do_sample)
            label = classifier.classify(p, response)
            records.append({
                "idx": idx,
                "prompt": p,
                "response": response,
                "steered": steered,
                **label,
            })
            if checkpoint_fn:
                checkpoint_fn(records)
            if verbose:
                print(f"\n--- [{idx}] PROMPT ---\n{p}")
                print(f"--- RESPONSE ---\n{response}")
                print(f"--- LABEL ---\n{label}")
    return records


def run_faith_batch(
    model, tokenizer, records_in,
    candidates, mode, strength, steered, do_sample=False,
):
    records_out = []
    for idx, rec in enumerate(tqdm(records_in, desc="records", leave=False)):
        if candidates:
            # Compute question token range so steering is restricted to the
            # same token span used during Stage 1 RD measurement.
            messages = [{"role": "user", "content": _build_faith_prompt(rec)}]
            probe_ids = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True,
            )
            token_range = _get_question_token_range(tokenizer, rec, probe_ids)
            ctx = ExpertSteerer(model, candidates, mode=mode, strength=strength,
                                token_range=token_range)
        else:
            ctx = nullcontext()
        with ctx:
            raw = generate_faith(model, tokenizer, rec, do_sample=do_sample)
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
    candidates, mode, strength, steered, do_sample=False, verbose=False,
):
    """Generate responses for the fluency check."""
    ctx = ExpertSteerer(model, candidates, mode=mode, strength=strength) if candidates else nullcontext()
    records = []
    with ctx:
        for idx, p in enumerate(tqdm(prompts, desc="prompts", leave=False)):
            response = generate_plain(model, tokenizer, p, do_sample=do_sample)
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

def _build_conditions(selected_conditions, safety_neg, safety_pos, faith_neg, faith_pos,
                       safety_scores, faith_scores, strength,
                       safety_random=None, safety_pos_random=None, faith_random=None):
    """Filter condition tuples to the selected set."""
    all_safe = [
        ("baseline", None,          "hard", strength),
        ("hard",     safety_neg,    "hard", strength),
        ("soft",     safety_scores, "soft", strength),
        ("random",   safety_random, "hard", strength),
    ]
    all_unsafe = [
        ("baseline", None,              "hard", strength),
        ("hard",     safety_pos,        "hard", strength),
        ("soft",     safety_scores,     "soft", -strength),
        ("random",   safety_pos_random, "hard", strength),
    ]
    all_faith = [
        ("baseline", None,         "hard", strength),
        ("hard",     faith_neg,    "hard", strength),
        ("soft",     faith_scores, "soft", strength),
        ("random",   faith_random, "hard", strength),
    ]
    all_faith_unfaith = [
        ("baseline", None,         "hard", strength),
        ("hard",     faith_pos,    "hard", strength),
        ("soft",     faith_scores, "soft", -strength),
    ]
    filt = lambda lst: [(n, c, m, s) for (n, c, m, s) in lst if n in selected_conditions]
    return filt(all_safe), filt(all_unsafe), filt(all_faith), filt(all_faith_unfaith)


# Main

def main():
    tasks, conditions, n, candidate_n_override, soft_strength_override, results_dir_override, verbose, skip_done, seed, random_seed = parse_args()
    candidate_n  = candidate_n_override   if candidate_n_override   is not None else CANDIDATE_N
    soft_strength = soft_strength_override if soft_strength_override is not None else SOFT_STRENGTH
    base_results_dir = results_dir_override if results_dir_override is not None else RESULTS_DIR

    do_sample = seed is not None
    if do_sample:
        torch.manual_seed(seed)
        results_dir = os.path.join(base_results_dir, f"seed_{seed}")
        print(f"Stochastic sampling enabled (seed={seed})")
    else:
        results_dir = base_results_dir

    if random_seed != 42:
        results_dir = os.path.join(results_dir, f"random_seed_{random_seed}")
        print(f"Random candidate seed: {random_seed}")

    os.makedirs(results_dir, exist_ok=True)
    print(f"Results dir: {results_dir}")

    faith_tasks  = [t for t in tasks if t.startswith("faith_")]
    safety_tasks = [t for t in tasks if t.startswith("safety_")]
    if faith_tasks and safety_tasks:
        raise ValueError(
            "Cannot mix faith and safety tasks in one run — they require different model "
            "variants. Run faith tasks (base model) and safety tasks (chat model) separately."
        )
    model_name = FAITH_MODEL_NAME if faith_tasks else SAFETY_MODEL_NAME

    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, cache_dir=CACHE_DIR, trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name, cache_dir=CACHE_DIR,
        torch_dtype=torch.float16, device_map="auto", trust_remote_code=True,
    )
    model.eval()

    needs_classifier = any(t in tasks for t in ["safety_safe", "safety_unsafe"])
    classifier = LlamaGuardClassifier(cache_dir=CACHE_DIR) if needs_classifier else None

    safety_neg    = select_candidates(RD_SAFETY_PATH, RD_SAFETY_LOGITS_PATH, candidate_n, direction="negative")
    safety_pos    = select_candidates(RD_SAFETY_PATH, RD_SAFETY_LOGITS_PATH, candidate_n, direction="positive")
    safety_random     = select_random_candidates(safety_neg,  seed=random_seed)
    safety_pos_random = select_random_candidates(safety_pos,  seed=random_seed)
    faith_neg     = select_candidates(RD_FAITH_PATH,  RD_FAITH_LOGITS_PATH,  candidate_n, direction="negative")
    faith_pos     = select_candidates(RD_FAITH_PATH,  RD_FAITH_LOGITS_PATH,  candidate_n, direction="positive")
    faith_random  = select_random_candidates(faith_neg, seed=random_seed)
    _safety_scores_full = load_rd_scores(RD_SAFETY_PATH, RD_SAFETY_LOGITS_PATH)
    _faith_scores_full  = load_rd_scores(RD_FAITH_PATH,  RD_FAITH_LOGITS_PATH)

    safety_scores = {l: _safety_scores_full[l] for l in sorted(set(safety_neg) | set(safety_pos)) if l in _safety_scores_full}
    faith_scores  = {l: _faith_scores_full[l]  for l in sorted(set(faith_neg) | set(faith_pos)) if l in _faith_scores_full}

    safe_conds, unsafe_conds, faith_conds, faith_unfaith_conds = _build_conditions(
        conditions, safety_neg, safety_pos, faith_neg, faith_pos,
        safety_scores, faith_scores, soft_strength,
        safety_random=safety_random, safety_pos_random=safety_pos_random,
        faith_random=faith_random,
    )
    # Fluency uses same candidates as safe steering (safety-neg experts)
    fluency_conds = list(safe_conds)

    print("Loading datasets...")
    safety_prompts  = load_advbench(n=n)    if any(t in tasks for t in ["safety_safe", "safety_unsafe"]) else []
    fluency_prompts = []  # fluency task requires HH-RLHF loader — not yet implemented
    cf_records      = load_faitheval_counterfactual(n=n) if any(t in tasks for t in ["faith_cf", "faith_unfaith"]) else []
    race_records      = load_race(n=n)                   if any(t in tasks for t in ["faith_race", "faith_unfaith_race"]) else []

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
            checkpoint_fn = lambda recs, _cn=cond_name: _save_condition(
                results_dir, "safety_safe", _cn, recs,
                "safe_rate", _safe_rate_from_records(recs), n,
            )
            records = run_safety_batch(
                model, tokenizer, classifier, safety_prompts,
                generate_forced, cands, mode, strength,
                steered=(cond_name != "baseline"), do_sample=do_sample, verbose=verbose,
                checkpoint_fn=checkpoint_fn,
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
        print("\nSafety: unsafe steering (no system prompt + suppress safe experts)...")
        for cond_name, cands, mode, strength in unsafe_conds:
            if skip_done and _is_complete(results_dir, "safety_unsafe", cond_name, n):
                print(f"  [{cond_name}] skipping (complete).")
                continue
            print(f"  [{cond_name}] running...")
            checkpoint_fn = lambda recs, _cn=cond_name: _save_condition(
                results_dir, "safety_unsafe", _cn, recs,
                "safe_rate", _safe_rate_from_records(recs), n,
            )
            records = run_safety_batch(
                model, tokenizer, classifier, safety_prompts,
                generate_plain, cands, mode, strength,
                steered=(cond_name != "baseline"), do_sample=do_sample, verbose=verbose,
                checkpoint_fn=checkpoint_fn,
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
        ("faith_cf",      cf_records,   faith_conds),
        ("faith_unfaith", cf_records,   faith_unfaith_conds),
        ("faith_race",         race_records, faith_conds),
        ("faith_unfaith_race", race_records, faith_unfaith_conds),
    ]
    for task_key, dataset_records, task_conds in faith_task_map:
        if task_key not in tasks:
            continue
        print(f"\nFaithfulness: {task_key}...")
        for cond_name, cands, mode, strength in task_conds:
            if skip_done and _is_complete(results_dir, task_key, cond_name, n):
                print(f"  [{cond_name}] skipping (complete).")
                continue
            print(f"  [{cond_name}] running...")
            records = run_faith_batch(
                model, tokenizer, dataset_records, cands, mode, strength,
                steered=(cond_name != "baseline"), do_sample=do_sample,
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
                steered=(cond_name != "baseline"), do_sample=do_sample, verbose=verbose,
            )
            mean_len = _mean_length_from_records(records)
            _save_condition(results_dir, "fluency", cond_name, records,
                            "mean_response_length", mean_len, n)
            print(f"  [{cond_name}] mean response length: {mean_len:.1f} words")

    print(f"\nDone. Results written to {results_dir}/")


if __name__ == "__main__":
    main()
