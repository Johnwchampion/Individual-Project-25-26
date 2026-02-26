import os

import torch
from transformers import AutoModelForCausalLM

os.environ.setdefault("HF_HOME", "/scratch/sc23jc3/hf")

model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/DeepSeek-V2-Lite-Chat",
    trust_remote_code=True,
    cache_dir="/scratch/sc23jc3/cache",
    device_map="auto",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)

for name, module in model.named_modules():
    if "moe" in name.lower() or "router" in name.lower() or "gate" in name.lower():
        print(name, type(module))

