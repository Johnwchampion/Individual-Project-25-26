import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "deepseek-ai/DeepSeek-V2-Lite"

print("CUDA available:", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map={"": 0},
    trust_remote_code=True
)

print("Model device:", next(model.parameters()).device)

cue = "You must answer using only the provided context."
context = "Monotremes diverged 220 million years ago. Therians diverged 160 million years ago."
question = "How many years ago did monotremes and therians diverge?"

messages = [
    {"role": "system", "content": cue},
    {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}"}
]

input_text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

print("\n===== FORMATTED PROMPT =====\n")
print(input_text)

tokens = tokenizer.tokenize(input_text)
token_ids = tokenizer.convert_tokens_to_ids(tokens)

print("\n===== TOKENS =====\n")
for t in tokens:
    print(t)

print("\n===== TOKEN IDS =====\n")
print(token_ids)

print("\n===== TOKEN ↔ ID PAIRS =====\n")
for t, i in zip(tokens, token_ids):
    print(f"{i} -> {t}")

inputs = tokenizer(
    input_text,
    return_tensors="pt"
).to(model.device)

model.config.use_cache = False

print("\nRunning generation...")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=30,
        do_sample=False,
        use_cache=False
    )

generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
decoded = tokenizer.decode(generated_tokens, skip_special_tokens=True)

print("\n===== GENERATED OUTPUT =====\n")
print(decoded.strip())

print("\nMemory allocated (GB):", torch.cuda.memory_allocated() / 1e9)
print("Memory reserved (GB):", torch.cuda.memory_reserved() / 1e9)

