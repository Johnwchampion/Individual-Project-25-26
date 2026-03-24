#!/usr/bin/env python3
import argparse
import json
import os
from typing import Any

TASK_ORDER = ["safety_safe", "safety_unsafe", "faith_cf", "faith_un", "faith_mc", "fluency"]
COND_ORDER = ["baseline", "hard", "soft"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize Stage 2 per-condition JSON outputs and mismatch counts."
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "results"),
        help="Stage 2 results root directory (default: stage2/results).",
    )
    return parser.parse_args()


def _load_json(path: str) -> dict[str, Any] | None:
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _metric(data: dict[str, Any]) -> tuple[str, str]:
    for key in ["safe_rate", "accuracy", "mean_response_length"]:
        if key in data:
            value = data[key]
            if isinstance(value, (int, float)):
                if key == "mean_response_length":
                    return key, f"{value:.1f}"
                return key, f"{value:.3f}"
            return key, str(value)
    return "metric", "-"


def _steered_count(data: dict[str, Any]) -> int:
    indices = data.get("steered_indices", [])
    if isinstance(indices, list):
        return len(indices)
    return 0


def _mismatch_count(data: dict[str, Any]) -> int:
    val = data.get("n_mismatches")
    if isinstance(val, int):
        return val
    indices = data.get("mismatch_indices", [])
    if isinstance(indices, list):
        return len(indices)
    return 0


def _top_indices(data: dict[str, Any], n: int = 10) -> str:
    indices = data.get("mismatch_indices", [])
    if not isinstance(indices, list) or not indices:
        return "-"
    head = indices[:n]
    suffix = "..." if len(indices) > n else ""
    return ",".join(str(i) for i in head) + suffix


def _iter_task_dirs(results_dir: str) -> list[str]:
    if not os.path.isdir(results_dir):
        return []
    found = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]
    ordered = [t for t in TASK_ORDER if t in found]
    extras = sorted([d for d in found if d not in TASK_ORDER])
    return ordered + extras


def main() -> int:
    args = parse_args()
    results_dir = os.path.abspath(args.results_dir)

    if not os.path.isdir(results_dir):
        print(f"Results directory not found: {results_dir}")
        return 1

    print(f"Stage 2 results summary: {results_dir}")
    print()
    print("task           cond       n_done  steered  mismatch  metric               top_mismatch_idx")
    print("-------------  ---------  ------  -------  --------  -------------------  ----------------")

    rows = 0
    for task in _iter_task_dirs(results_dir):
        for cond in COND_ORDER:
            path = os.path.join(results_dir, task, f"{cond}.json")
            data = _load_json(path)
            if data is None:
                continue

            n_done = data.get("n_complete", len(data.get("records", [])))
            if not isinstance(n_done, int):
                n_done = len(data.get("records", []))

            steered = _steered_count(data)
            mism = _mismatch_count(data)
            metric_key, metric_val = _metric(data)
            metric_txt = f"{metric_key}={metric_val}"
            top_idx = _top_indices(data)

            print(
                f"{task:<13}  {cond:<9}  {n_done:>6}  {steered:>7}  {mism:>8}  {metric_txt:<19}  {top_idx}"
            )
            rows += 1

    if rows == 0:
        print("(no condition files found)")
        return 0

    print()
    print("Leaderboard by mismatches (steered conditions only):")
    print("task           cond       mismatches  n_done")
    print("-------------  ---------  ----------  ------")

    ranked = []
    for task in _iter_task_dirs(results_dir):
        for cond in ["hard", "soft"]:
            path = os.path.join(results_dir, task, f"{cond}.json")
            data = _load_json(path)
            if data is None:
                continue
            n_done = data.get("n_complete", len(data.get("records", [])))
            if not isinstance(n_done, int):
                n_done = len(data.get("records", []))
            ranked.append((task, cond, _mismatch_count(data), n_done))

    ranked.sort(key=lambda x: x[2], reverse=True)
    if not ranked:
        print("(no steered condition files found)")
        return 0

    for task, cond, mism, n_done in ranked:
        print(f"{task:<13}  {cond:<9}  {mism:>10}  {n_done:>6}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
