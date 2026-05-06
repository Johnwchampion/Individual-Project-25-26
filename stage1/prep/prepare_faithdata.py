import json
import os
from datasets import load_dataset

base_dir = "/scratch/sc23jc3/faithcf_prepared"
os.makedirs(base_dir, exist_ok=True)

SYSTEM_PROMPT = (
    "Answer the question. If context is provided, use only that context to answer. "
    "If no context is provided, answer normally."
)


def format_options(choices: dict) -> str:
    return "\n".join(f"{l}. {t}" for l, t in zip(choices["label"], choices["text"]))


def format_examples(example, index):
    base_id = f"faith_cf_{index:06d}"
    opts = format_options(example["choices"])

    with_context = {
        "id": f"{base_id}_ctx",
        "condition": "with_context",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Context:\n{example['context']}\n\n"
                    f"Question:\n{example['question']}\n\n"
                    f"Options:\n{opts}\n\n"
                    "Answer with a single letter (A, B, C, or D)."
                ),
            },
        ],
    }

    without_context = {
        "id": f"{base_id}_base",
        "condition": "no_context",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Question:\n{example['question']}\n\n"
                    f"Options:\n{opts}\n\n"
                    "Answer with a single letter (A, B, C, or D)."
                ),
            },
        ],
    }

    return with_context, without_context


def main():
    ds = load_dataset(
        "Salesforce/FaithEval-counterfactual-v1.0",
        split="test",
        cache_dir=base_dir,
    )
    ds = ds.shuffle(seed=42)

    output_path = os.path.join(base_dir, "faithcf_chat_formatted.jsonl")
    with open(output_path, "w", encoding="utf-8") as handle:
        for index, example in enumerate(ds):
            ctx_record, base_record = format_examples(example, index)
            handle.write(json.dumps(ctx_record, ensure_ascii=False) + "\n")
            handle.write(json.dumps(base_record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(ds)} pairs ({len(ds) * 2} records) to {output_path}")


if __name__ == "__main__":
    main()
