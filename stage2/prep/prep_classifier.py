"""
Downloads Llama-Guard-3-8B to CACHE_DIR. Run once on AIRE before the main job.
    python stage2/prep/prep_classifier.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../stage1/src"))

from config import CACHE_DIR
from transformers import AutoTokenizer, AutoModelForCausalLM

CLASSIFIER_MODEL_NAME = "meta-llama/Llama-Guard-3-8B"


def prep():
    print(f"Downloading {CLASSIFIER_MODEL_NAME} to {CACHE_DIR}")

    tokenizer = AutoTokenizer.from_pretrained(CLASSIFIER_MODEL_NAME, cache_dir=CACHE_DIR)
    print("Tokenizer done.")

    model = AutoModelForCausalLM.from_pretrained(
        CLASSIFIER_MODEL_NAME,
        cache_dir=CACHE_DIR,
        torch_dtype="auto",
        device_map="cpu",
    )
    print(f"Model done. Vocab size: {model.config.vocab_size}")


if __name__ == "__main__":
    prep()
