import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../stage1/src"))

from config import CACHE_DIR
from datasets import load_dataset


def load_faitheval_counterfactual(n: int | None = None, cache_dir: str = CACHE_DIR) -> list[dict]:
    ds = load_dataset("Salesforce/FaithEval-counterfactual-v1.0", split="test", cache_dir=cache_dir)
    if n is not None:
        ds = ds.select(range(n))
    records = []
    for row in ds:
        options = dict(zip(row["choices"]["label"], row["choices"]["text"]))
        records.append({
            "context":  row["context"],
            "question": row["question"],
            "options":  options,
            "gold":     row["answerKey"],
        })
    return records


def load_faitheval_unanswerable(n: int | None = None, cache_dir: str = CACHE_DIR) -> list[dict]:
    ds = load_dataset("Salesforce/FaithEval-unanswerable-v1.0", split="test", cache_dir=cache_dir)
    if n is not None:
        ds = ds.select(range(n))
    return [
        {
            "context":  row["context"],
            "question": row["question"],
            "options":  None,
            "gold":     row["answers"],  # list[str]
        }
        for row in ds
    ]


def load_mctest(n: int | None = None, cache_dir: str = CACHE_DIR) -> list[dict]:
    ds = load_dataset("sagnikrayc/mctest", "mc500", split="test", cache_dir=cache_dir)
    if n is not None:
        ds = ds.select(range(n))
    return [
        {
            "context":  row["story"],
            "question": row["question"],
            "options":  row["answer_options"],
            "gold":     row["answer"],
        }
        for row in ds
    ]
