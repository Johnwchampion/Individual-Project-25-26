import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM



def load_model(model_name, cache_dir):
    os.makedirs(cache_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        cache_dir=cache_dir
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        cache_dir=cache_dir
    )

    model.config.use_cache = False
    return model, tokenizer


def generate(model, tokenizer, messages, max_new_tokens=50):
    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=False
        )

    return outputs
