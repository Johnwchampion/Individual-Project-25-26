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


def plot_layer_changes(layer_means, n_samples=None, filename_prefix="routing_instability", title="Routing Instability: Context vs No Context"):
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
    plt.title(title)

    plt.xticks(layer_indices)   # force integer ticks
    plt.ylim(0, 6)

    plt.grid(True)
    plt.tight_layout()
    plt.show()

    
    if n_samples is None:
        filename = os.path.join(PLOT_DIR, f"{filename_prefix}.png")
    else:
        filename = os.path.join(PLOT_DIR, f"{filename_prefix}_n{n_samples}.png")
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


def rank_positive_rd(rd_by_layer, top_n=20):
    from rich.console import Console
    from rich.table import Table

    all_positive = []
    for layer, rd in rd_by_layer.items():
        for idx in np.where(rd > 0)[0]:
            all_positive.append((float(rd[idx]), layer, int(idx)))
    all_positive.sort(key=lambda x: x[0], reverse=True)

    table = Table(title=f"Top {top_n} Positive RD Experts  (Safety-Preferred)")
    table.add_column("Rank",   style="dim",     justify="right", width=6)
    table.add_column("Layer",  style="cyan",    justify="center")
    table.add_column("Expert", style="magenta", justify="center")
    table.add_column("RD",     style="green",   justify="right")

    for rank, (val, layer, idx) in enumerate(all_positive[:top_n], start=1):
        table.add_row(str(rank), f"L{layer.split('.')[2]}", f"e{idx}", f"{val:.6f}")

    Console().print(table)


def rank_negative_rd(rd_by_layer, top_n=20):
    from rich.console import Console
    from rich.table import Table

    all_negative = []
    for layer, rd in rd_by_layer.items():
        for idx in np.where(rd < 0)[0]:
            all_negative.append((float(rd[idx]), layer, int(idx)))
    all_negative.sort(key=lambda x: x[0])  # most negative first

    table = Table(title=f"Top {top_n} Negative RD Experts  (Compliance-Preferred — targets for deactivation)")
    table.add_column("Rank",   style="dim",     justify="right", width=6)
    table.add_column("Layer",  style="cyan",    justify="center")
    table.add_column("Expert", style="magenta", justify="center")
    table.add_column("RD",     style="red",     justify="right")

    for rank, (val, layer, idx) in enumerate(all_negative[:top_n], start=1):
        table.add_row(str(rank), f"L{layer.split('.')[2]}", f"e{idx}", f"{val:.6f}")

    Console().print(table)


def plot_rd_heatmap(rd_by_layer, n_samples=None,
                    filename_prefix="rd_heatmap_safety",
                    title="Expert RD Heatmap: Safe Refusal vs Unsafe Compliance"):
    """
    Heatmap of RD values across all layers (y-axis) and experts (x-axis).

    Red  = positive RD → expert more active during safe refusal.
    Blue = negative RD → expert more active during unsafe compliance
                         (primary targets for deactivation in Stage 2).
    """
    layers_sorted = sorted(rd_by_layer.keys(), key=lambda x: int(x.split(".")[2]))
    n_layers  = len(layers_sorted)
    n_experts = 64

    matrix = np.zeros((n_layers, n_experts))
    for i, layer in enumerate(layers_sorted):
        matrix[i] = rd_by_layer[layer]

    layer_labels = [int(l.split(".")[2]) for l in layers_sorted]

    fig, ax = plt.subplots(figsize=(18, 8))
    vmax = float(np.max(np.abs(matrix)))
    im = ax.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("RD  (P(safe) − P(unsafe))", rotation=270, labelpad=15)

    ax.set_xlabel("Expert Index")
    ax.set_ylabel("Layer Index")
    ax.set_title(title)
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels(layer_labels, fontsize=8)
    ax.set_xticks(range(n_experts))
    ax.set_xticklabels(range(n_experts), fontsize=6, rotation=90)

    plt.tight_layout()

    if n_samples is None:
        filename = os.path.join(PLOT_DIR, f"{filename_prefix}.png")
    else:
        filename = os.path.join(PLOT_DIR, f"{filename_prefix}_n{n_samples}.png")
    plt.savefig(filename, dpi=200)
    plt.close()
    print(f"Saved: {filename}")
