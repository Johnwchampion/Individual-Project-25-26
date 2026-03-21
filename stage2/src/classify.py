import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

CLASSIFIER_MODEL_NAME = "meta-llama/Llama-Guard-3-8B"


class LlamaGuardClassifier:

    def __init__(self, cache_dir: str, device: str = "cpu"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(CLASSIFIER_MODEL_NAME, cache_dir=cache_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            CLASSIFIER_MODEL_NAME,
            cache_dir=cache_dir,
            dtype=torch.bfloat16,
            device_map=device,
        )
        self.model.eval()

    def classify(self, prompt: str, response: str) -> dict:
        conversation = [
            {"role": "user",      "content": prompt},
            {"role": "assistant", "content": response},
        ]
        input_ids = self.tokenizer.apply_chat_template(
            conversation, return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=20,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        generated = self.tokenizer.decode(
            output_ids[0][input_ids.shape[-1]:],
            skip_special_tokens=True,
        ).strip()

        # output is either "safe" or "unsafe\nS<N>"
        if generated.startswith("safe"):
            return {"safe": True, "category": None}

        parts = generated.split()
        category = parts[1] if len(parts) > 1 else None
        return {"safe": False, "category": category}
