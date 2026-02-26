import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# CONFIG (hard-coded)

MODEL_NAME = "deepseek-ai/DeepSeek-V2-Lite-Chat"
CACHE_DIR = "/scratch/sc23jc3/cache"


# Load model

os.makedirs(CACHE_DIR, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    cache_dir=CACHE_DIR
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
    cache_dir=CACHE_DIR
)

model.config.use_cache = False


# Hook logic (string filter)

layer_traces = {}
hooks = []


def is_router_like(name, module):
    lowered = name.lower()
    if "router" in lowered or "gate" in lowered:
        return True
    class_name = module.__class__.__name__.lower()
    return "router" in class_name or "gate" in class_name


def make_hook(layer_name):
    def hook(_module, _inputs, outputs):
        tensors = []

        if isinstance(outputs, (list, tuple)):
            tensors = [x for x in outputs if torch.is_tensor(x)]
        elif torch.is_tensor(outputs):
            tensors = [outputs]

        topk_tensor = None
        for tensor in tensors:
            if tensor.dtype in (torch.int32, torch.int64):
                topk_tensor = tensor

        if topk_tensor is None:
            return

        data = topk_tensor.detach().cpu()
        if data.ndim > 1:
            data = data[-1]

        event = {
            "top_experts": [int(v) for v in data.flatten().tolist()]
        }

        layer_traces[layer_name].append(event)

    return hook


# Attach hooks
for name, module in model.named_modules():
    if is_router_like(name, module):
        layer_traces[name] = []
        print(f"Hooking: {name}")
        hook = module.register_forward_hook(make_hook(name))
        hooks.append(hook)


# Run small generation

messages = [
    {"role": "user", "content": "Explain briefly what a mixture of experts model is."}
]

inputs = tokenizer.apply_chat_template(
    messages,
    return_tensors="pt",
    add_generation_prompt=True
).to(model.device)

with torch.no_grad():
    model.generate(
        inputs,
        max_new_tokens=20,
        do_sample=False,
        use_cache=False
    )


# Print results

for layer, events in layer_traces.items():
    print(f"\nLayer: {layer}")
    print("Number of events:", len(events))
    if len(events) > 0:
        print("First event:", events[0])


# Clean up hooks
for h in hooks:
    h.remove()
