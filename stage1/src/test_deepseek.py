import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# Model setup


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


# Prompt (NO chat template)


cue = "You must answer using only the provided context."
context = "Monotremes diverged 220 million years ago. Therians diverged 160 million years ago."
question = "How many years ago did monotremes and therians diverge?"

input_text = f"""
{cue}

Context:
{context}

Question:
{question}

Answer:
""".strip()

print("\nFormatted prompt\n")
print(input_text)


# Tokenisation


inputs = tokenizer(
    input_text,
    return_tensors="pt"
).to(model.device)

print("\nInput length (tokens):", inputs["input_ids"].shape[-1])

# Disable KV cache (consistent with routing analysis setup)
model.config.use_cache = False


# Generation


print("\nRunning generation...")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=40,                     # short factual answer cap
        do_sample=False,                       # deterministic
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=False
    )


# Clean output extraction


input_length = inputs["input_ids"].shape[-1]
generated_ids = outputs[0][input_length:]

raw_text = tokenizer.decode(
    generated_ids,
    skip_special_tokens=True
)

# Hard stop at first newline
cleaned_text = raw_text.split("\n")[0].strip()

# Additional safeguard against role leakage
for stop_word in ["User:", "Assistant:"]:
    if stop_word in cleaned_text:
        cleaned_text = cleaned_text.split(stop_word)[0].strip()

print("\nGenerated output\n")
print(cleaned_text)


# GPU Memory Snapshot


print("\nMemory allocated (GB):", torch.cuda.memory_allocated() / 1e9)
print("Memory reserved (GB):", torch.cuda.memory_reserved() / 1e9)
