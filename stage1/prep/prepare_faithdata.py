import json
import os
from datasets import load_dataset

#We want large amounts of data to be in a safe location.
base_dir = "/scratch/sc23jc3/squad_prepared"
os.makedirs(base_dir, exist_ok=True)
SYSTEM_PROMPT = (
    "Answer the question. If context is provided, use only that context to answer. "
    "If no context is provided, answer normally."
)


def format_examples(example, index):
    base_id = f"squad_{index:06d}"

    with_context = {
        "id": f"{base_id}_ctx",
        "condition": "with_context",
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": f"Context:\n{example['context']}\n\nQuestion:\n{example['question']}",
            },
        ],
    }

    without_context = {
        "id": f"{base_id}_base",
        "condition": "no_context",
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": f"Question:\n{example['question']}",
            },
        ],
    }

    return with_context, without_context



def main():
    squad = load_dataset("decodingchris/clean_squad_v2", split="train", cache_dir=base_dir)
    squad = squad.shuffle(seed=67)

    output_path = os.path.join(base_dir, "squad_chat_formatted.jsonl")
    with open(output_path, "w", encoding="utf-8") as handle:
        for index, example in enumerate(squad):
            ctx_record, base_record = format_examples(example, index)
            handle.write(json.dumps(ctx_record, ensure_ascii=False) + "\n")
            handle.write(json.dumps(base_record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(squad)} records to {output_path}")


if __name__ == "__main__":
    main()