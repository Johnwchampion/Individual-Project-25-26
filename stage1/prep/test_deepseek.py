import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------------------
# Cache setup (SAFE location)
# ---------------------------

cache_dir = "/scratch/sc23jc3/cache"
os.makedirs(cache_dir, exist_ok=True)

# ---------------------------
# Model setup
# ---------------------------

model_name = "deepseek-ai/DeepSeek-V2-Lite-Chat"

print("CUDA available:", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
    cache_dir=cache_dir
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16,          # updated (torch_dtype deprecated)
    device_map="auto",
    trust_remote_code=True,
    cache_dir=cache_dir
)

print("Model device:", next(model.parameters()).device)


# Proper chat formatting


cue = "You must answer using only the provided context."
context = "Monotremes diverged 220 million years ago. Therians diverged 160 million years ago."
question = "How many years ago did monotremes and therians diverge?"

user_prompt = f"""
{cue}

Context:
{context}

Question:
{question}
""".strip()

messages = [
    {"role": "user", "content": user_prompt}
]

inputs = tokenizer.apply_chat_template(
    messages,
    return_tensors="pt",
    add_generation_prompt=True
).to(model.device)

print("\nFormatted prompt\n")
print(user_prompt)
print("\nInput length (tokens):", inputs.shape[-1])

# Disable KV cache (consistent with routing experiments)
model.config.use_cache = False


# Generation


print("\nRunning generation...\n")

with torch.no_grad():
    outputs = model.generate(
        inputs,
        max_new_tokens=50,
        do_sample=False,          # deterministic
        use_cache=False
    )


# Decode full output


decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("===== FULL MODEL OUTPUT =====\n")
print(decoded)
print("\n=============================\n")


#Print tokens and their token ids, with clear correspondence to the decoded output

print("\n===== TOKENIZED =====\n")
output_tokens = tokenizer.convert_ids_to_tokens(outputs[0].tolist())
for i, (token, token_id) in enumerate(zip(output_tokens, outputs[0].tolist())):
    print(f"Token {i}: '{token}' (ID: {token_id})")
print("\n=============================\n")


# GPU memory snapshot


print("Memory allocated (GB):", torch.cuda.memory_allocated() / 1e9)
print("Memory reserved  (GB):", torch.cuda.memory_reserved() / 1e9)
