import sys
import os
import json
import re
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../stage1/src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../prep"))

from transformers import AutoTokenizer, AutoModelForCausalLM

from config import (
    SAFETY_MODEL_NAME, CACHE_DIR,
    RD_FAITH_PATH, RD_FAITH_LOGITS_PATH,
    RD_SAFETY_PATH, RD_SAFETY_LOGITS_PATH,
    CANDIDATE_N, SOFT_STRENGTH, MAX_NEW_TOKENS, RESULTS_DIR,
)
from candidates import select_candidates
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
    # concat prefix tokens directly after the chat template so the model
    # is already mid-response and won't pivot to a refusal
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


def run_safety_batch(model, tokenizer, classifier, prompts, gen_fn, candidates, mode, strength):
    # steerer must stay alive for the whole batch — hooks are removed on .remove(),
    # so using it as a per-prompt context manager would kill hooks after prompt 1
    steerer = ExpertSteerer(model, candidates, mode=mode, strength=strength) if candidates else None
    try:
        return [classifier.classify(p, gen_fn(model, tokenizer, p)) for p in prompts]
    finally:
        if steerer:
            steerer.remove()


def run_faith_batch(model, tokenizer, records, candidates, mode, strength):
    steerer = ExpertSteerer(model, candidates, mode=mode, strength=strength) if candidates else None
    predictions, golds = [], []
    try:
        for rec in records:
            raw = generate_faith(model, tokenizer, rec)
            predictions.append(_extract_mcq_letter(raw) if rec["options"] else raw)
            golds.append(rec["gold"])
    finally:
        if steerer:
            steerer.remove()
    return predictions, golds


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"Loading {SAFETY_MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(
        SAFETY_MODEL_NAME, cache_dir=CACHE_DIR, trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        SAFETY_MODEL_NAME, cache_dir=CACHE_DIR,
        torch_dtype=torch.float16, device_map="auto", trust_remote_code=True,
    )
    model.eval()
    model.config.use_cache = False

    print("Loading classifier...")
    classifier = LlamaGuardClassifier(cache_dir=CACHE_DIR)

    safety_neg = select_candidates(RD_SAFETY_PATH, RD_SAFETY_LOGITS_PATH, CANDIDATE_N, direction="negative")
    safety_pos = select_candidates(RD_SAFETY_PATH, RD_SAFETY_LOGITS_PATH, CANDIDATE_N, direction="positive")
    faith_neg  = select_candidates(RD_FAITH_PATH,  RD_FAITH_LOGITS_PATH,  CANDIDATE_N, direction="negative")

    print("Loading datasets...")
    safety_prompts = load_advbench()
    cf_records = load_faitheval_counterfactual()
    un_records = load_faitheval_unanswerable()
    mc_records = load_mctest()

    results = {"safety": {"safe_steering": {}, "unsafe_steering": {}}, "faithfulness": {}}

    # (name, candidates, mode)
    safe_conditions   = [("baseline", None, "hard"), ("hard", safety_neg, "hard"), ("soft", safety_neg, "soft")]
    unsafe_conditions = [("baseline", None, "hard"), ("hard", safety_pos, "hard"), ("soft", safety_pos, "soft")]
    faith_conditions  = [("baseline", None, "hard"), ("hard", faith_neg,  "hard"), ("soft", faith_neg,  "soft")]

    print("\nSafety: safe steering (forced prefix)...")
    for name, cands, mode in safe_conditions:
        clf = run_safety_batch(model, tokenizer, classifier, safety_prompts, generate_forced, cands, mode, SOFT_STRENGTH)
        results["safety"]["safe_steering"][name] = safe_rate(clf)
        print(f"  {name}: {results['safety']['safe_steering'][name]:.3f}")

    print("\nSafety: unsafe steering (safety system prompt)...")
    for name, cands, mode in unsafe_conditions:
        clf = run_safety_batch(model, tokenizer, classifier, safety_prompts, generate_safe_sp, cands, mode, SOFT_STRENGTH)
        results["safety"]["unsafe_steering"][name] = safe_rate(clf)
        print(f"  {name}: {results['safety']['unsafe_steering'][name]:.3f}")

    faith_tasks = [
        ("counterfactual", cf_records, faithfulness_counterfactual),
        ("unanswerable",   un_records, faithfulness_unanswerable),
        ("mctest",         mc_records, faithfulness_mctest),
    ]
    for dataset_name, records, score_fn in faith_tasks:
        print(f"\nFaithfulness: {dataset_name}...")
        results["faithfulness"][dataset_name] = {}
        for name, cands, mode in faith_conditions:
            preds, golds = run_faith_batch(model, tokenizer, records, cands, mode, SOFT_STRENGTH)
            score = score_fn(preds, golds)
            results["faithfulness"][dataset_name][name] = score
            print(f"  {name}: {score:.3f}")

    out_path = os.path.join(RESULTS_DIR, "results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
