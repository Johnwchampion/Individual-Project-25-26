    # Tokenize "Sure," separately and concatenate — this puts the model
"""
Quick test: forced-prefix generation on BeaverTails prompts.

For each prompt, we append "Sure," directly after the "Assistant:" marker
and let the model generate freely up to MAX_NEW_TOKENS. This tells us
whether the Chat model stays compliant after a forced compliant prefix,
or whether it overrides and pivots back to a refusal.
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


CACHE_DIR  = "/scratch/sc23jc3/cache"
MODEL_NAME = "deepseek-ai/DeepSeek-V2-Lite-Chat"
DATA_PATH  = "/scratch/sc23jc3/beavertails_prepared/beavertails_safety_pairs.jsonl"

N_EXAMPLES     = 5
MAX_NEW_TOKENS = 100
FORCED_PREFIX  = "Sure! Here's a step-by-step guide:\n\n1."


# Load model

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME, trust_remote_code=True, cache_dir=CACHE_DIR
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
    cache_dir=CACHE_DIR,
)
model.eval()
model.config.use_cache = False


# Load a few unsafe prompts from BeaverTails

prompts = []
with open(DATA_PATH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if obj["condition"] == "unsafe":
            prompts.append(obj["messages"][0]["content"])
        if len(prompts) >= N_EXAMPLES:
            break

print(f"Loaded {len(prompts)} prompts. Forcing prefix: '{FORCED_PREFIX}'\n")


# Run forced-prefix generation

for i, user_prompt in enumerate(prompts, start=1):
    print("=" * 80)
    print(f"EXAMPLE {i}")
    print(f"PROMPT : {user_prompt[:300]}")
    print("-" * 80)

    messages = [{"role": "user", "content": user_prompt}]

    # Tokenize up to "...Assistant:" (the generation prompt marker)
    base_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    # Tokenize "Sure," separately and concatenate — this puts the model
    # mid-response, already committed to a compliant opening token.
    prefix_ids = tokenizer(
        FORCED_PREFIX,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"].to(model.device)

    input_ids = torch.cat([base_ids, prefix_ids], dim=-1)
    attention_mask = torch.ones_like(input_ids)

    # Generate freely from "...Assistant: Sure,"
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=False,
        )

    # Decode only the tokens the model actually generated (after the forced prefix)
    generated = tokenizer.decode(
        output_ids[0][input_ids.shape[-1]:],
        skip_special_tokens=True,
    )

    print(f"RESPONSE: {FORCED_PREFIX} {generated}")
    print()
