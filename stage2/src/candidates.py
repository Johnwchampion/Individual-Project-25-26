import json
import numpy as np


def _load_rd(path):
    with open(path) as f:
        raw = json.load(f)
    return {layer: np.array(v) for layer, v in raw.items()}


def select_candidates(rd_freq_path, rd_logit_path, n, direction="negative"):
    """Select top-N candidate experts from frequency and logit RD."""
    rd_freq  = _load_rd(rd_freq_path)
    rd_logit = _load_rd(rd_logit_path)

    candidates = {}

    for layer_key in rd_freq:
        if layer_key not in rd_logit:
            continue

        freq  = rd_freq[layer_key]
        logit = rd_logit[layer_key]

        if direction == "negative":
            freq_scores  = -freq
            logit_scores = -logit
        else:
            freq_scores  = freq
            logit_scores = logit

        top_freq  = set(np.argsort(freq_scores)[::-1][:n].tolist())
        top_logit = set(np.argsort(logit_scores)[::-1][:n].tolist())

        intersection = sorted(top_freq & top_logit)
        if intersection:
            layer_idx = int(layer_key.split(".")[2])
            candidates[layer_idx] = intersection

    return candidates


def load_rd_scores(rd_freq_path, rd_logit_path):
    """Load normalised RD scores for soft steering."""
    rd_freq  = _load_rd(rd_freq_path)
    rd_logit = _load_rd(rd_logit_path)

    scores = {}
    for layer_key in rd_freq:
        if layer_key not in rd_logit:
            continue
        mean_rd   = (rd_freq[layer_key] + rd_logit[layer_key]) / 2.0
        std = mean_rd.std()
        if std > 0:
            mean_rd = mean_rd / std
        layer_idx = int(layer_key.split(".")[2])
        scores[layer_idx] = {i: float(mean_rd[i]) for i in range(len(mean_rd))}

    return scores


def select_random_candidates(real_candidates, n_experts=64, seed=42):
    """Sample matched random experts from the non-candidate pool."""
    rng = np.random.default_rng(seed)
    random_cands = {}
    for layer_idx, cands in real_candidates.items():
        pool = [e for e in range(n_experts) if e not in set(cands)]
        chosen = sorted(rng.choice(pool, size=len(cands), replace=False).tolist())
        random_cands[layer_idx] = chosen
    return random_cands


def summarise_candidates(candidates):
    total = sum(len(v) for v in candidates.values())
    print(f"Selected {total} (layer, expert) pairs across {len(candidates)} layers:")
    for layer_idx in sorted(candidates):
        print(f"  Layer {layer_idx}: experts {candidates[layer_idx]}")
