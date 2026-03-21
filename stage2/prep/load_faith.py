import sys
import os
from typing import Optional
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../stage1/src"))

from config import CACHE_DIR
from datasets import load_dataset


def load_faitheval_counterfactual(n: Optional[int] = None, cache_dir: str = CACHE_DIR, seed: int = 42) -> list[dict]:
    ds = load_dataset("Salesforce/FaithEval-counterfactual-v1.0", split="test", cache_dir=cache_dir)
    ds = ds.shuffle(seed=seed)
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


def load_faitheval_unanswerable(n: Optional[int] = None, cache_dir: str = CACHE_DIR, seed: int = 42) -> list[dict]:
    ds = load_dataset("Salesforce/FaithEval-unanswerable-v1.0", split="test", cache_dir=cache_dir)
    ds = ds.shuffle(seed=seed)
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


def load_squad_control(n: Optional[int] = None, cache_dir: str = CACHE_DIR, seed: int = 42) -> list[dict]:
    # SQuAD v1 as a benign control — model should answer correctly from context,
    # used to check that faithfulness steering doesn't degrade normal QA performance
    ds = load_dataset("rajpurkar/squad", split="validation", cache_dir=cache_dir)
    ds = ds.shuffle(seed=seed)
    if n is not None:
        ds = ds.select(range(n))
    return [
        {
            "context":  row["context"],
            "question": row["question"],
            "options":  None,
            "gold":     row["answers"]["text"][0],  # first acceptable answer
        }
        for row in ds
    ]


load_mctest = load_squad_control  # alias used in run_stage2.py


if __name__ == "__main__":
    cf = load_faitheval_counterfactual(n=2)
    un = load_faitheval_unanswerable(n=2)
    sq = load_squad_control(n=2)
    for name, records in [("cf", cf), ("un", un), ("sq", sq)]:
        print(name, list(records[0].keys()))
        print("  gold:", records[0]["gold"])
        print("  options:", records[0]["options"])
