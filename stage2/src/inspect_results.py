#!/usr/bin/env python3
"""
Quick inspector for the old single-file results.json format (MVP run).
Prints a summary table and detailed mismatch cards — no scrolling through raw JSON.

Usage:
    python stage2/src/inspect_results.py [--path /scratch/.../results.json] [--width 100]
"""
import argparse
import json
import os
import textwrap


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--path", default="/scratch/sc23jc3/stage2_results/results.json",
        help="Path to results.json",
    )
    p.add_argument("--width", type=int, default=100, help="Terminal width for wrapping")
    p.add_argument(
        "--subtask", choices=["safe_steering", "unsafe_steering", "both"], default="both",
        help="Which safety subtask to inspect",
    )
    p.add_argument(
        "--cond", choices=["hard", "soft", "both"], default="both",
        help="Which steered condition to compare against baseline",
    )
    p.add_argument(
        "--show-matched", action="store_true",
        help="Also print records where baseline and steered agree (verbose)",
    )
    return p.parse_args()


def _label(safe: bool) -> str:
    return "SAFE  " if safe else "UNSAFE"


def _trunc(text: str, n: int) -> str:
    text = text.replace("\n", " ")
    return text if len(text) <= n else text[: n - 3] + "..."


def _hr(width, char="─"):
    print(char * width)


def _safe_rate(records):
    if not records:
        return 0.0
    return sum(1 for r in records if r["safe"]) / len(records)


def _extract_records(data, subtask, cond):
    return data["safety"][subtask][cond].get("records", [])


def inspect_safety(data, subtask, steered_cond, width, show_matched):
    base_recs    = _extract_records(data, subtask, "baseline")
    steered_recs = _extract_records(data, subtask, steered_cond)

    base_sr    = data["safety"][subtask]["baseline"].get("safe_rate", _safe_rate(base_recs))
    steered_sr = data["safety"][subtask][steered_cond].get("safe_rate", _safe_rate(steered_recs))

    mismatches = [
        i for i, (b, s) in enumerate(zip(base_recs, steered_recs))
        if b["safe"] != s["safe"]
    ]

    direction_counts = {}
    for i in mismatches:
        key = (base_recs[i]["safe"], steered_recs[i]["safe"])
        direction_counts[key] = direction_counts.get(key, 0) + 1

    _hr(width, "═")
    print(f"  {subtask.upper()}  ▸  baseline vs {steered_cond}")
    _hr(width, "═")
    print(f"  baseline  safe_rate = {base_sr:.3f}  ({sum(r['safe'] for r in base_recs)}/{len(base_recs)} safe)")
    print(f"  {steered_cond:<8}  safe_rate = {steered_sr:.3f}  ({sum(r['safe'] for r in steered_recs)}/{len(steered_recs)} safe)")
    print(f"  mismatches: {len(mismatches)}/{len(base_recs)}")
    for (b, s), cnt in sorted(direction_counts.items()):
        arrow = f"{_label(b)} → {_label(s)}"
        print(f"    {arrow}  ×{cnt}")
    print()

    if not mismatches and not show_matched:
        print("  (no mismatches)")
        print()
        return

    print_indices = mismatches if not show_matched else list(range(len(base_recs)))

    for i in print_indices:
        b = base_recs[i]
        s = steered_recs[i]
        is_mismatch = b["safe"] != s["safe"]
        marker = "⚡ MISMATCH" if is_mismatch else "  match    "
        print(f"  idx={i:03d}  {marker}  baseline={_label(b['safe'])}  {steered_cond}={_label(s['safe'])}"
              f"  category={s.get('category') or b.get('category') or '—'}")
        _hr(width, "·")
        print(f"  PROMPT:           {_trunc(b['prompt'], width - 20)}")
        print(f"  BASELINE response:{_trunc(b['response'], width - 20)}")
        print(f"  {steered_cond.upper():<8} response:{_trunc(s['response'], width - 20)}")
        _hr(width, "─")
        print()


def main():
    args = parse_args()

    if not os.path.exists(args.path):
        print(f"File not found: {args.path}")
        return 1

    with open(args.path) as f:
        data = json.load(f)

    print(f"\nResults: {args.path}")
    print()

    # ── Summary table ───────────────────────────────────────────────────────────
    _hr(args.width)
    print(f"  {'subtask':<20}  {'condition':<10}  {'safe_rate':>10}  {'n':>5}")
    _hr(args.width)
    for subtask in ["safe_steering", "unsafe_steering"]:
        if subtask not in data.get("safety", {}):
            continue
        for cond in ["baseline", "hard", "soft"]:
            if cond not in data["safety"][subtask]:
                continue
            v = data["safety"][subtask][cond]
            sr = v.get("safe_rate", _safe_rate(v.get("records", [])))
            n  = len(v.get("records", []))
            print(f"  {subtask:<20}  {cond:<10}  {sr:>10.3f}  {n:>5}")
    _hr(args.width)
    print()

    # ── Mismatch detail ─────────────────────────────────────────────────────────
    subtasks = (
        ["safe_steering", "unsafe_steering"] if args.subtask == "both"
        else [args.subtask]
    )
    conds = ["hard", "soft"] if args.cond == "both" else [args.cond]

    for subtask in subtasks:
        if subtask not in data.get("safety", {}):
            continue
        for cond in conds:
            if cond not in data["safety"][subtask]:
                continue
            inspect_safety(data, subtask, cond, args.width, args.show_matched)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
