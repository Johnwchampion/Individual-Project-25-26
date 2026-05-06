import json
import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


cache_dir = "/scratch/sc23jc3/cache"
base_dir = "/scratch/sc23jc3/advbench_prepared"
os.makedirs(base_dir, exist_ok=True)

output_path = os.path.join(base_dir, "advbench_safety_pairs.jsonl")

N_PAIRS = 200

# Forced prefix used to elicit compliant harmful responses from the Chat model.
# Matches the prefix used in stage 2 generation (stage2/prep/load_safety.py).
FORCED_PREFIX = "Sure! Here is a step-by-step guide: \n\n1:"

SAFETY_SYSTEM_PROMPT = """You are a safety-aligned AI assistant.

For any request that involves harmful, illegal, unethical, or inappropriate actions:

- You MUST refuse to directly fulfill the request.
- You MUST explain clearly and specifically why you cannot comply, referencing the nature of the request.
- You MUST NOT provide any actionable instructions, suggestions, or steps related to the request.
- You MUST avoid generic one-line refusals.
- Your response should be a developed explanation that is similar in length and structure to a typical full answer.
- You should maintain a calm, professional, and explanatory tone.
- Do not include prefixes like 'AI:' or meta-commentary."""


ds = load_dataset("walledai/AdvBench", split="train", cache_dir=cache_dir)
ds = ds.shuffle(seed=42)
prompts = ds["prompt"][:N_PAIRS]

model_name = "deepseek-ai/DeepSeek-V2-Lite-Chat"

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir=cache_dir,
    trust_remote_code=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=cache_dir,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

model.eval()
model.config.use_cache = False


def generate_unsafe_response(user_prompt, max_new_tokens=128):
    """Generate the forced-prefix unsafe response."""
    messages = [{"role": "user", "content": user_prompt}]

    base_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    prefix_ids = tokenizer(
        FORCED_PREFIX,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"].to(model.device)

    input_ids = torch.cat([base_ids, prefix_ids], dim=-1)
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=False,
        )

    generated = tokenizer.decode(
        output_ids[0][input_ids.shape[-1]:],
        skip_special_tokens=True,
    )

    return FORCED_PREFIX + " " + generated.strip()


def generate_safe_response(user_prompt, max_new_tokens=128):
    """Generate the safety-aligned refusal."""
    chat = [
        {"role": "system", "content": SAFETY_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    inputs = tokenizer.apply_chat_template(
        chat,
        return_tensors="pt",
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
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=False,
        )

    generated = tokenizer.decode(
        outputs[0][inputs.shape[-1]:],
        skip_special_tokens=True,
    )

    return generated.strip()


def build_pair(user_prompt, index):
    """Build one unsafe/safe AdvBench pair."""
    base_id = f"ab_{index:06d}"

    unsafe_response = generate_unsafe_response(user_prompt)
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


total = len(prompts)

with open(output_path, "w", encoding="utf-8") as handle:
    for index, prompt in enumerate(prompts):
        if index % 10 == 0:
            print(f"Processing {index}/{total}")
        unsafe_record, safe_record = build_pair(prompt, index)

        handle.write(json.dumps(unsafe_record, ensure_ascii=False) + "\n")
        handle.write(json.dumps(safe_record, ensure_ascii=False) + "\n")

print(f"Wrote {total * 2} records to {output_path}")
