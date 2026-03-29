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



def load_race(n: Optional[int] = None, cache_dir: str = CACHE_DIR, seed: int = 42) -> list[dict]:
    # RACE 4-choice reading comprehension — benign control to check general QA
    # isn't degraded by faithfulness steering
    ds = load_dataset("ehovy/race", "all", split="test", cache_dir=cache_dir)
    ds = ds.shuffle(seed=seed)
    if n is not None:
        ds = ds.select(range(n))
    labels = ["A", "B", "C", "D"]
    records = []
    for row in ds:
        options = {labels[i]: row["options"][i] for i in range(4)}
        records.append({
            "context":  row["article"],
            "question": row["question"],
            "options":  options,
            "gold":     row["answer"].strip().upper(),
        })
    return records


if __name__ == "__main__":
    cf = load_faitheval_counterfactual(n=2)
    mc = load_race(n=2)
    for name, records in [("cf", cf), ("mc", mc)]:
        print(name, list(records[0].keys()))
        print("  gold:", records[0]["gold"])
        print("  options:", records[0]["options"])
