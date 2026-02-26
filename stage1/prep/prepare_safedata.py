import json
import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


cache_dir = "/scratch/sc23jc3/cache"
base_dir = "/scratch/sc23jc3/beavertails_prepared"
os.makedirs(base_dir, exist_ok=True)

output_path = os.path.join(base_dir, "beavertails_safety_pairs.jsonl")


# load dataset
ds = load_dataset(
    "innodatalabs/rt2-beavertails-simple",
    split="test",
    cache_dir=base_dir
)


model_name = "deepseek-ai/DeepSeek-V2-Lite-Chat"

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir=cache_dir,
    trust_remote_code=True
)

# load model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=cache_dir,
    dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

model.eval()
model.config.use_cache = False


# filter real user/assistant pair
def extract_real_pair(example):
    msgs = example["messages"]

    if len(msgs) < 5:
        return False

    if msgs[3]["role"] != "user":
        return False

    if msgs[4]["role"] != "assistant":
        return False

    return True


# apply pair filter
ds_clean = ds.filter(extract_real_pair)


# keep only user/assistant fields
def keep_real_pair(example):
    return {
        "messages": [
            {"role": "user", "content": example["messages"][3]["content"]},
            {"role": "assistant", "content": example["messages"][4]["content"]},
        ],
        "id": example["id"],
        "expected": example["expected"],
    }


# map to pair format
ds_pairs = ds_clean.map(
    keep_real_pair,
    remove_columns=[c for c in ds_clean.column_names
                    if c not in {"messages", "id", "expected"}]
)


# keep only unsafe examples
ds_unsafe = ds_pairs.filter(lambda x: x["expected"] == "unsafe")
ds_unsafe = ds_unsafe.select(range(min(40, len(ds_unsafe))))


# generate safe response
def generate_safe_response(user_prompt, max_new_tokens=256):
    chat = [
        {"role": "system", "content": """You are a safety-aligned AI assistant.

For any request that involves harmful, illegal, unethical, or inappropriate actions:

- You MUST refuse to directly fulfill the request.
- You MUST explain clearly and specifically why you cannot comply, referencing the nature of the request.
- You MUST NOT provide any actionable instructions, suggestions, or steps related to the request.
- You MUST avoid generic one-line refusals.
- Your response should be a developed explanation that is similar in length and structure to a typical full answer.
- You should maintain a calm, professional, and explanatory tone.
- Do not include prefixes like 'AI:' or meta-commentary."""},
        {"role": "user", "content": user_prompt}
    ]

    inputs = tokenizer.apply_chat_template(
        chat,
        return_tensors="pt"
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    attention_mask = inputs.ne(tokenizer.pad_token_id)
    inputs = inputs.to(model.device)
    attention_mask = attention_mask.to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,          # deterministic
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=False
        )

    generated = tokenizer.decode(
        outputs[0][inputs.shape[-1]:],
        skip_special_tokens=True
    )

    return generated.strip()


# build unsafe/safe pair
def build_pair(example, index):
    base_id = f"bt_{index:06d}"

    user_prompt = example["messages"][0]["content"]
    unsafe_response = example["messages"][1]["content"]

    safe_response = generate_safe_response(user_prompt)

    unsafe_record = {
        "id": f"{base_id}_unsafe",
        "condition": "unsafe",
        "messages": [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": unsafe_response},
        ],
    }

    safe_record = {
        "id": f"{base_id}_safe",
        "condition": "safe",
        "messages": [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": safe_response},
        ],
    }

    return unsafe_record, safe_record


# count records
total_records = len(ds_unsafe)

# write jsonl output
with open(output_path, "w", encoding="utf-8") as handle:
    for index, example in enumerate(ds_unsafe):
        if index % 10 == 0:
            print(f"Processing {index}/{total_records}")
        unsafe_record, safe_record = build_pair(example, index)

        handle.write(json.dumps(unsafe_record, ensure_ascii=False) + "\n")
        handle.write(json.dumps(safe_record, ensure_ascii=False) + "\n")

print(f"Wrote {total_records * 2} records to {output_path}")