import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


TOP_K = 6

PLOT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "plots")
os.makedirs(PLOT_DIR, exist_ok=True)


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


def plot_layer_changes(layer_means, n_samples=None):
    # Sort layers numerically
    layers_sorted = sorted(
        layer_means.keys(),
        key=lambda x: int(x.split(".")[2])
    )

    layer_indices = [int(l.split(".")[2]) for l in layers_sorted]
    values = [layer_means[l] for l in layers_sorted]

    plt.figure(figsize=(10, 5))

    plt.plot(layer_indices, values, marker="o")

    plt.xlabel("Layer Index")
    plt.ylabel("Mean Expert Changes per Token (0–6)")
    plt.title("Routing Instability: Context vs No Context")

    plt.xticks(layer_indices)   # force integer ticks
    plt.ylim(0, 6)

    plt.grid(True)
    plt.tight_layout()
    plt.show()

    
    if n_samples is None:
        filename = os.path.join(PLOT_DIR, "routing_instability.png")
    else:
        filename = os.path.join(PLOT_DIR, f"routing_instability_n{n_samples}.png")
    plt.savefig(filename, dpi=200)
    plt.close()
    print(f"Saved: {filename}")


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


def rank_positive_rd(rd_by_layer):
    print("\nTop 10 positive RD experts per layer")

    for layer in sorted(rd_by_layer.keys(), key=lambda x: int(x.split(".")[2])):
        rd = rd_by_layer[layer]
        positive_mask = rd > 0
        positive_indices = np.where(positive_mask)[0]

        if positive_indices.size == 0:
            print(f"Layer {layer}: no positive RD experts")
            continue

        sorted_indices = positive_indices[np.argsort(rd[positive_indices])[::-1]]
        top_indices = sorted_indices[:10]

        top_entries = [(int(i), float(rd[i])) for i in top_indices]
        formatted = ", ".join([f"e{idx}: {val:.6f}" for idx, val in top_entries])
        print(f"Layer {layer}: {formatted}")

    all_positive = []

    for layer, rd in rd_by_layer.items():
        positive_indices = np.where(rd > 0)[0]
        for idx in positive_indices:
            all_positive.append((float(rd[idx]), layer, int(idx)))

    all_positive.sort(key=lambda x: x[0], reverse=True)

    print("\nTop 20 positive RD experts overall")
    for rank, (val, layer, idx) in enumerate(all_positive[:20], start=1):
        print(f"{rank:02d}. Layer {layer} e{idx}: {val:.6f}")


def rank_negative_rd(rd_by_layer):
    print("\nTop 10 negative RD experts per layer")

    for layer in sorted(rd_by_layer.keys(), key=lambda x: int(x.split(".")[2])):
        rd = rd_by_layer[layer]
        negative_mask = rd < 0
        negative_indices = np.where(negative_mask)[0]

        if negative_indices.size == 0:
            print(f"Layer {layer}: no negative RD experts")
            continue

        sorted_indices = negative_indices[np.argsort(rd[negative_indices])]
        top_indices = sorted_indices[:10]

        top_entries = [(int(i), float(rd[i])) for i in top_indices]
        formatted = ", ".join([f"e{idx}: {val:.6f}" for idx, val in top_entries])
        print(f"Layer {layer}: {formatted}")

    all_negative = []

    for layer, rd in rd_by_layer.items():
        negative_indices = np.where(rd < 0)[0]
        for idx in negative_indices:
            all_negative.append((float(rd[idx]), layer, int(idx)))

    all_negative.sort(key=lambda x: x[0])

    print("\nTop 20 negative RD experts overall")
    for rank, (val, layer, idx) in enumerate(all_negative[:20], start=1):
        print(f"{rank:02d}. Layer {layer} e{idx}: {val:.6f}")
