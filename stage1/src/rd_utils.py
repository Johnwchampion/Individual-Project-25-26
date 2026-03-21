import os
import json
import numpy as np

TOP_K = 6


def compute_layer_token_differences(routing_with, routing_without, return_counts=False):
    layer_results = {}
    for layer in routing_with.keys():
        if layer not in routing_without:
            continue
        flat_with = routing_with[layer]
        flat_without = routing_without[layer]
        T = len(flat_with) // TOP_K
        total_diff = 0
        for t in range(T):
            start = t * TOP_K
            end = start + TOP_K
            set_with = set(flat_with[start:end])
            set_without = set(flat_without[start:end])
            overlap = len(set_with.intersection(set_without))
            diff = TOP_K - overlap
            total_diff += diff
        if return_counts:
            layer_results[layer] = (total_diff, T)
        else:
            layer_results[layer] = total_diff / T
    return layer_results


def accumulate_expert_counts(question_routing, count_dict, token_dict):
    for layer, flat in question_routing.items():
        if not flat:
            continue
        counts = np.bincount(np.array(flat, dtype=np.int64), minlength=64)
        T = len(flat) // TOP_K
        if layer not in count_dict:
            count_dict[layer] = np.zeros(64, dtype=np.int64)
        count_dict[layer] += counts
        token_dict[layer] += T


def accumulate_mean_logits(question_logits, logit_sum_dict, token_dict):
    for layer, token_logits in question_logits.items():
        if not token_logits:
            continue
        arr = np.array(token_logits, dtype=np.float32)
        if layer not in logit_sum_dict:
            logit_sum_dict[layer] = np.zeros(arr.shape[1], dtype=np.float64)
        logit_sum_dict[layer] += arr.sum(axis=0)
        token_dict[layer] += arr.shape[0]


def compute_rd(count_with, tokens_with, count_without, tokens_without):
    rd_by_layer = {}
    for layer in count_with.keys():
        if layer not in count_without:
            continue
        T_with = tokens_with.get(layer, 0)
        T_without = tokens_without.get(layer, 0)
        if T_with == 0 or T_without == 0:
            continue
        p_with = count_with[layer] / T_with
        p_without = count_without[layer] / T_without
        if not np.isclose(p_with.sum(), TOP_K, rtol=1e-3, atol=1e-3):
            print(f"Warning: p_with sum != {TOP_K} for layer {layer}: {p_with.sum():.4f}")
        if not np.isclose(p_without.sum(), TOP_K, rtol=1e-3, atol=1e-3):
            print(f"Warning: p_without sum != {TOP_K} for layer {layer}: {p_without.sum():.4f}")
        rd_by_layer[layer] = p_with - p_without
    return rd_by_layer


def compute_rd_logits(logit_sum_a, tokens_a, logit_sum_b, tokens_b):
    rd_by_layer = {}
    for layer in logit_sum_a:
        if layer not in logit_sum_b:
            continue
        T_a = tokens_a.get(layer, 0)
        T_b = tokens_b.get(layer, 0)
        if T_a == 0 or T_b == 0:
            continue
        mean_a = logit_sum_a[layer] / T_a
        mean_b = logit_sum_b[layer] / T_b
        rd_by_layer[layer] = mean_a - mean_b
    return rd_by_layer


def save_rd(rd_by_layer, filepath):
    serialisable = {layer: rd.tolist() for layer, rd in rd_by_layer.items()}
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(serialisable, f)
    print(f"Saved RD to {filepath}")


def load_rd(filepath):
    with open(filepath) as f:
        raw = json.load(f)
    return {layer: np.array(v) for layer, v in raw.items()}
