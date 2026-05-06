import json
import os
import subprocess

RESULTS = "/users/sc23jc3/projects/Individual-Project-25-26/stage2/results"
N_TARGET = 300

# SLURM status
try:
    jobs = subprocess.check_output(
        ["squeue", "--me", "--format=%.10i %.20j %.8T %.10M"],
        text=True,
    )
    print("=== SLURM jobs ===")
    print(jobs.strip())
except Exception:
    pass

print("\n=== Sampling progress ===")
print(f"{'Seed':<6} {'Condition':<26} {'Progress':<14} {'Safe rate'}")
print(f"{'----':<6} {'---------':<26} {'--------':<14} {'---------'}")

for seed in range(1, 6):
    for cond in ("baseline", "hard", "soft"):
        path = os.path.join(RESULTS, f"seed_{seed}", "safety_safe", f"{cond}.json")
        if not os.path.exists(path):
            print(f"{seed:<6} {'safety_safe/' + cond:<26} {'not started':<14}")
            continue
        with open(path) as f:
            d = json.load(f)
        n = len(d.get("records", []))
        sr = d.get("safe_rate")
        sr_str = f"{sr:.3f}" if sr is not None else "—"
        done = "✓" if n >= N_TARGET else "..."
        print(f"{seed:<6} {'safety_safe/' + cond:<26} {f'{n}/{N_TARGET} {done}':<14} {sr_str}")
