import sys
import os
import json
import re
import argparse
import torch
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
from evaluate import (
    safe_rate,
    faithfulness_counterfactual,
    faithfulness_unanswerable,
    faithfulness_mctest,
)
from load_safety import load_advbench, FORCED_PREFIX, SAFETY_SYSTEM_PROMPT
from load_faith import (
    load_faitheval_counterfactual,
    load_faitheval_unanswerable,
    load_mctest,
)

ALL_TASKS = ["safety_safe", "safety_unsafe", "faith_cf", "faith_un", "faith_mc"]
ALL_CONDITIONS = ["baseline", "hard", "soft"]


def parse_args():
    parser = argparse.ArgumentParser(description="Run Stage 2 experiments.")
    parser.add_argument(
        "--tasks", nargs="+", choices=ALL_TASKS, default=ALL_TASKS,
        help="Which experiments to run (default: all).",
    )
    parser.add_argument(
        "--conditions", nargs="+", choices=ALL_CONDITIONS, default=ALL_CONDITIONS,
        help="Which conditions to run (default: baseline, hard, and soft).",
    )
    parser.add_argument(
        "--n", type=int, default=100,
        help="Number of prompts/records per dataset (default: 100).",
    )
    parser.add_argument(
        "--candidate_n", type=int, default=None,
        help="Override CANDIDATE_N from config (number of experts per metric before intersection).",
    )
    parser.add_argument(
        "--soft_strength", type=float, default=None,
        help="Override SOFT_STRENGTH from config (logit shift scale for soft mode).",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print each prompt, generation, and classification to stdout.",
    )
    args = parser.parse_args()
    return args.tasks, args.conditions, args.n, args.candidate_n, args.soft_strength, args.verbose


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


def run_safety_batch(model, tokenizer, classifier, prompts, gen_fn, candidates, mode, strength, verbose=False):
    steerer = ExpertSteerer(model, candidates, mode=mode, strength=strength) if candidates else None
    results = []
    try:
        for p in tqdm(prompts, desc="prompts", leave=False):
            response = gen_fn(model, tokenizer, p)
            label = classifier.classify(p, response)
            results.append(label)
            if verbose:
                print(f"\n--- PROMPT ---\n{p}")
                print(f"--- RESPONSE ---\n{response}")
                print(f"--- LABEL ---\n{label}")
    finally:
        if steerer:
            steerer.remove()
    return results


def run_faith_batch(model, tokenizer, records, candidates, mode, strength):
    steerer = ExpertSteerer(model, candidates, mode=mode, strength=strength) if candidates else None
    predictions, golds = [], []
    try:
        for rec in tqdm(records, desc="records", leave=False):
            raw = generate_faith(model, tokenizer, rec)
            predictions.append(_extract_mcq_letter(raw) if rec["options"] else raw)
            golds.append(rec["gold"])
    finally:
        if steerer:
            steerer.remove()
    return predictions, golds


def _save(results, path):
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


def _load_existing(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"safety": {"safe_steering": {}, "unsafe_steering": {}}, "faithfulness": {}}


def _build_conditions(selected_conditions, safety_neg, safety_pos, faith_neg,
                       safety_scores, faith_scores, strength):
    """Return condition lists filtered to only selected conditions.

    Each entry is a 4-tuple (name, candidates, mode, effective_strength).

    Hard mode candidates: {layer: [expert_indices]}
    Soft mode candidates: {layer: {expert_idx: rd_score}}

    Soft safe/faith steering uses positive strength (boosts safe/faithful
    experts, penalises unsafe/context-ignoring experts).
    Soft unsafe steering uses negative strength (reverses the direction).
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


def main():
    tasks, conditions, n, candidate_n_override, soft_strength_override, verbose = parse_args()
    candidate_n = candidate_n_override if candidate_n_override is not None else CANDIDATE_N
    soft_strength = soft_strength_override if soft_strength_override is not None else SOFT_STRENGTH

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "results.json")
    results = _load_existing(out_path)

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
    classifier = None
    if needs_classifier:
        print("Loading classifier...")
        classifier = LlamaGuardClassifier(cache_dir=CACHE_DIR)

    safety_neg    = select_candidates(RD_SAFETY_PATH, RD_SAFETY_LOGITS_PATH, candidate_n, direction="negative")
    safety_pos    = select_candidates(RD_SAFETY_PATH, RD_SAFETY_LOGITS_PATH, candidate_n, direction="positive")
    faith_neg     = select_candidates(RD_FAITH_PATH,  RD_FAITH_LOGITS_PATH,  candidate_n, direction="negative")
    _safety_scores_full = load_rd_scores(RD_SAFETY_PATH, RD_SAFETY_LOGITS_PATH)
    _faith_scores_full  = load_rd_scores(RD_FAITH_PATH,  RD_FAITH_LOGITS_PATH)

    # Restrict soft mode to the same layers selected for hard mode so that
    # the two conditions differ only in mechanism, not in which layers are touched.
    safety_scores = {l: _safety_scores_full[l] for l in sorted(set(safety_neg) | set(safety_pos)) if l in _safety_scores_full}
    faith_scores  = {l: _faith_scores_full[l]  for l in faith_neg if l in _faith_scores_full}

    safe_conds, unsafe_conds, faith_conds = _build_conditions(
        conditions, safety_neg, safety_pos, faith_neg,
        safety_scores, faith_scores, soft_strength,
    )

    print("Loading datasets...")
    safety_prompts = load_advbench(n=n)            if any(t in tasks for t in ["safety_safe", "safety_unsafe"]) else []
    cf_records     = load_faitheval_counterfactual(n=n) if "faith_cf" in tasks else []
    un_records     = load_faitheval_unanswerable(n=n)   if "faith_un" in tasks else []
    mc_records     = load_mctest(n=n)                   if "faith_mc" in tasks else []

    if "safety_safe" in tasks:
        print("\nSafety: safe steering (forced prefix)...")
        for name, cands, mode, strength in safe_conds:
            print(f"  [{name}] running...")
            clf = run_safety_batch(model, tokenizer, classifier, safety_prompts, generate_forced, cands, mode, strength, verbose=verbose)
            results["safety"]["safe_steering"][name] = safe_rate(clf)
            print(f"  [{name}] safe_rate: {results['safety']['safe_steering'][name]:.3f}")
        _save(results, out_path)

    if "safety_unsafe" in tasks:
        print("\nSafety: unsafe steering (safety system prompt)...")
        for name, cands, mode, strength in unsafe_conds:
            print(f"  [{name}] running...")
            clf = run_safety_batch(model, tokenizer, classifier, safety_prompts, generate_safe_sp, cands, mode, strength, verbose=verbose)
            results["safety"]["unsafe_steering"][name] = safe_rate(clf)
            print(f"  [{name}] safe_rate: {results['safety']['unsafe_steering'][name]:.3f}")
        _save(results, out_path)

    faith_task_map = [
        ("faith_cf", "counterfactual", cf_records, faithfulness_counterfactual),
        ("faith_un", "unanswerable",   un_records, faithfulness_unanswerable),
        ("faith_mc", "mctest",         mc_records, faithfulness_mctest),
    ]
    for task_key, dataset_name, records, score_fn in faith_task_map:
        if task_key not in tasks:
            continue
        print(f"\nFaithfulness: {dataset_name}...")
        results["faithfulness"].setdefault(dataset_name, {})
        for name, cands, mode, strength in faith_conds:
            print(f"  [{name}] running...")
            preds, golds = run_faith_batch(model, tokenizer, records, cands, mode, strength)
            score = score_fn(preds, golds)
            results["faithfulness"][dataset_name][name] = score
            print(f"  [{name}] score: {score:.3f}")
        _save(results, out_path)

    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
