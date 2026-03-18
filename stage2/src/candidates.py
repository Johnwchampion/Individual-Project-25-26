import json
import numpy as np


def _load_rd(path):
    with open(path) as f:
        raw = json.load(f)
    return {layer: np.array(v) for layer, v in raw.items()}


def select_candidates(rd_freq_path, rd_logit_path, n, direction="negative"):
    """
    Returns {layer_index (int): [expert_indices]} for experts that appear in
    the top-N by |RD| in the given direction on BOTH frequency and logit metrics.

    direction="negative": suppress experts preferred in condition B
                          (unsafe compliance / context-ignoring)
    direction="positive": suppress experts preferred in condition A
                          (safe refusal / context-faithful)
    """
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


def summarise_candidates(candidates):
    total = sum(len(v) for v in candidates.values())
    print(f"Selected {total} (layer, expert) pairs across {len(candidates)} layers:")
    for layer_idx in sorted(candidates):
        print(f"  Layer {layer_idx}: experts {candidates[layer_idx]}")
