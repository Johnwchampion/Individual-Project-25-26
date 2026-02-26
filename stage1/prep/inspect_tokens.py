import json
from itertools import islice
from transformers import AutoTokenizer

DATA_PATH = "/scratch/sc23jc3/squad_prepared/squad_chat_formatted.jsonl"
MODEL_NAME = "deepseek-ai/DeepSeek-V2-Lite-Chat"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    cache_dir="/scratch/sc23jc3/cache"
)

def inspect_tokens(n_pairs=2):
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        records = [json.loads(next(f)) for _ in range(n_pairs * 2)]

    for i, record in enumerate(records):
        print("=" * 120)
        print(f"RECORD {i+1}")
        print("ID:", record["id"])
        print("Condition:", record["condition"])
        print()

        # Apply chat template
        input_ids = tokenizer.apply_chat_template(
            record["messages"],
            add_generation_prompt=True,
            return_tensors="pt"
        )[0]

        tokens = tokenizer.convert_ids_to_tokens(input_ids.tolist())

        print("Total tokens:", len(tokens))
        print()

        for idx, (tok, tid) in enumerate(zip(tokens, input_ids.tolist())):
            print(f"{idx:03d} | {repr(tok):30} | {tid}")

        print("\n")

inspect_tokens(n_pairs=2)
