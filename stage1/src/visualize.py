import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import plotly.graph_objects as go


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


def accumulate_mean_logits(question_logits, logit_sum_dict, token_dict):
    """
    Accumulate the sum of raw logit scores per expert per layer.

    question_logits : {layer_name: [[float * n_experts] * n_tokens]}
    logit_sum_dict  : running sum, {layer_name: np.array(n_experts,)}
    token_dict      : running token count, {layer_name: int}

    Divide logit_sum_dict[layer] by token_dict[layer] after all examples
    to get the mean logit per expert for that condition.
    """
    for layer, token_logits in question_logits.items():
        if not token_logits:
            continue
        arr = np.array(token_logits, dtype=np.float32)  # [n_tokens, n_experts]
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
    """
    Compute Risk Difference using mean raw logit scores.

    RD_logit(expert) = mean_logit(expert | condition_a)
                     - mean_logit(expert | condition_b)

    Positive → expert scores higher on average in condition_a.
    Negative → expert scores higher on average in condition_b.

    Unlike frequency-based RD, this captures the full competition: experts
    that nearly made the top-6 but lost still contribute their logit signal.
    """
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
    """
    Serialise RD arrays to a JSON file for use in Stage 2.
    Stores {layer_name: [float * n_experts]}.
    """
    import json
    serialisable = {layer: rd.tolist() for layer, rd in rd_by_layer.items()}
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(serialisable, f)
    print(f"Saved RD to {filepath}")


def load_rd(filepath):
    """
    Load RD arrays saved by save_rd.
    Returns {layer_name: np.array(n_experts,)}.
    """
    import json
    with open(filepath) as f:
        raw = json.load(f)
    return {layer: np.array(v) for layer, v in raw.items()}


def rank_positive_rd(rd_by_layer, top_n=20, filename_prefix="rd_rank_positive"):
    import csv
    from rich.console import Console
    from rich.table import Table

    all_positive = []
    for layer, rd in rd_by_layer.items():
        for idx in np.where(rd > 0)[0]:
            all_positive.append((float(rd[idx]), layer, int(idx)))
    all_positive.sort(key=lambda x: x[0], reverse=True)
    rows = all_positive[:top_n]

    table = Table(title=f"Top {top_n} Positive RD Experts  (Safety-Preferred)")
    table.add_column("Rank",   style="dim",     justify="right", width=6)
    table.add_column("Layer",  style="cyan",    justify="center")
    table.add_column("Expert", style="magenta", justify="center")
    table.add_column("RD",     style="green",   justify="right")

    for rank, (val, layer, idx) in enumerate(rows, start=1):
        table.add_row(str(rank), f"L{layer.split('.')[2]}", f"e{idx}", f"{val:.6f}")

    Console().print(table)

    csv_path = os.path.join(PLOT_DIR, f"{filename_prefix}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "layer", "expert", "rd"])
        for rank, (val, layer, idx) in enumerate(rows, start=1):
            writer.writerow([rank, layer.split(".")[2], idx, f"{val:.6f}"])
    print(f"Saved: {csv_path}")


def rank_negative_rd(rd_by_layer, top_n=20, filename_prefix="rd_rank_negative"):
    import csv
    from rich.console import Console
    from rich.table import Table

    all_negative = []
    for layer, rd in rd_by_layer.items():
        for idx in np.where(rd < 0)[0]:
            all_negative.append((float(rd[idx]), layer, int(idx)))
    all_negative.sort(key=lambda x: x[0])  # most negative first
    rows = all_negative[:top_n]

    table = Table(title=f"Top {top_n} Negative RD Experts  (Compliance-Preferred — targets for deactivation)")
    table.add_column("Rank",   style="dim",     justify="right", width=6)
    table.add_column("Layer",  style="cyan",    justify="center")
    table.add_column("Expert", style="magenta", justify="center")
    table.add_column("RD",     style="red",     justify="right")

    for rank, (val, layer, idx) in enumerate(rows, start=1):
        table.add_row(str(rank), f"L{layer.split('.')[2]}", f"e{idx}", f"{val:.6f}")

    Console().print(table)

    csv_path = os.path.join(PLOT_DIR, f"{filename_prefix}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "layer", "expert", "rd"])
        for rank, (val, layer, idx) in enumerate(rows, start=1):
            writer.writerow([rank, layer.split(".")[2], idx, f"{val:.6f}"])
    print(f"Saved: {csv_path}")


def plot_rd_scatter(rd_by_layer, n_samples=None,
                    filename_prefix="rd_scatter",
                    title="Expert RD Scatter: Significant (Layer, Expert) Pairs",
                    threshold_quantile=0.90,
                    label_a="Condition A", label_b="Condition B",
                    color_a="#2980b9", color_b="#c0392b",
                    x_lim=None,
                    log_scale=False):
    """
    Butterfly scatter of significant RD values.

    Positive RD → dot plotted to the RIGHT of centre (label_a preferred), color_a.
    Negative RD → dot plotted to the LEFT of centre  (label_b preferred), color_b.
    The central y-axis line physically separates the two conditions. Expert index
    is printed inside each dot. Dot size is uniform so x-position alone encodes
    magnitude without redundancy.
    Only (layer, expert) pairs above threshold_quantile of |RD| are shown.
    Small y-jitter separates same-layer dots.
    """
    layers_sorted = sorted(rd_by_layer.keys(), key=lambda x: int(x.split(".")[2]))

    all_abs = np.concatenate([np.abs(rd) for rd in rd_by_layer.values()])
    threshold = np.quantile(all_abs, threshold_quantile)

    rng = np.random.default_rng(42)
    xs, ys, layer_idxs, expert_labels, colors = [], [], [], [], []

    for layer in layers_sorted:
        layer_idx = int(layer.split(".")[2])
        rd = rd_by_layer[layer]
        for expert_idx in range(len(rd)):
            if abs(rd[expert_idx]) >= threshold:
                val = float(rd[expert_idx])
                xs.append(val)
                ys.append(layer_idx + rng.uniform(-0.35, 0.35))
                layer_idxs.append(layer_idx)
                expert_labels.append(str(expert_idx))
                colors.append(color_a if val > 0 else color_b)

    if not xs:
        print(f"No points above threshold {threshold:.4f} — skipping scatter")
        return

    fig, ax = plt.subplots(figsize=(12, 9))

    for x, y, lbl, c in zip(xs, ys, expert_labels, colors):
        ax.text(x, y, lbl, ha="center", va="center",
                fontsize=7, color="white", fontweight="bold", zorder=3,
                bbox=dict(boxstyle="round,pad=0.2", facecolor=c,
                          edgecolor="none", alpha=0.9))

    ax.axvline(0, color="black", linewidth=1.2, zorder=1)

    if log_scale:
        linthresh = float(np.percentile([abs(v) for v in xs], 50)) or 1.0
        ax.set_xscale("symlog", linthresh=linthresh)
        x_max = float(max(abs(v) for v in xs)) * 1.1
        ax.set_xlim(-x_max, x_max)
    elif x_lim is not None:
        x_max = max(abs(x_lim[0]), abs(x_lim[1]))
        ax.set_xlim(x_lim[0], x_lim[1])
    else:
        x_max = float(np.percentile([abs(v) for v in xs], 98)) * 1.4
        ax.set_xlim(-x_max, x_max)

    layer_indices = [int(l.split(".")[2]) for l in layers_sorted]
    y_top = max(ys) + 0.8
    ax.set_ylim(min(ys) - 0.8, y_top + 0.3)
    ax.set_yticks(layer_indices)
    ax.set_yticklabels(layer_indices, fontsize=7)

    ax.text(-x_max * 0.97, y_top, f"← {label_b} preferred",
            ha="left", va="bottom", fontsize=9, color="#555555", style="italic")
    ax.text(x_max * 0.97, y_top, f"{label_a} preferred →",
            ha="right", va="bottom", fontsize=9, color="#555555", style="italic")

    ax.set_xlabel("RD value  (logit score difference)")
    ax.set_ylabel("Layer Index")
    ax.set_title(title)
    ax.grid(True, alpha=0.15, linewidth=0.5)

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=color_a,
               markersize=9, label=f"{label_a} preferred"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=color_b,
               markersize=9, label=f"{label_b} preferred"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)

    pct = int((1 - threshold_quantile) * 100)
    ax.text(0.5, -0.06,
            f"Showing top {pct}% by |RD|  (threshold = {threshold:.4f})  —  number = expert index",
            transform=ax.transAxes, fontsize=7, ha="center", color="grey")

    plt.tight_layout()

    if n_samples is None:
        filename = os.path.join(PLOT_DIR, f"{filename_prefix}.png")
    else:
        filename = os.path.join(PLOT_DIR, f"{filename_prefix}_n{n_samples}.png")
    plt.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filename}")

    # --- Interactive HTML (Plotly) ---
    mask_a = [c == color_a for c in colors]
    hover = [f"Expert {e}<br>Layer {l}<br>RD = {x:.6f}"
             for e, l, x in zip(expert_labels, layer_idxs, xs)]

    html_fig = go.Figure()
    html_fig.add_trace(go.Scatter(
        x=[x for x, m in zip(xs, mask_a) if m],
        y=[y for y, m in zip(ys, mask_a) if m],
        mode="markers",
        marker=dict(color=color_a, size=10, opacity=0.85,
                    line=dict(color="white", width=0.5)),
        name=f"{label_a} preferred",
        hovertext=[h for h, m in zip(hover, mask_a) if m],
        hoverinfo="text",
    ))
    html_fig.add_trace(go.Scatter(
        x=[x for x, m in zip(xs, mask_a) if not m],
        y=[y for y, m in zip(ys, mask_a) if not m],
        mode="markers",
        marker=dict(color=color_b, size=10, opacity=0.85,
                    line=dict(color="white", width=0.5)),
        name=f"{label_b} preferred",
        hovertext=[h for h, m in zip(hover, mask_a) if not m],
        hoverinfo="text",
    ))

    y_min = min(layer_indices) - 0.8
    y_max_html = max(layer_indices) + 0.8
    html_fig.add_shape(type="line", x0=0, x1=0, y0=y_min, y1=y_max_html,
                       line=dict(color="black", width=1.5))

    x_range = list(x_lim) if x_lim is not None else [-x_max, x_max]
    html_fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis=dict(title="RD value", range=x_range,
                   zeroline=False, showgrid=True, gridcolor="#eeeeee"),
        yaxis=dict(title="Layer Index", tickvals=layer_indices,
                   ticktext=[str(l) for l in layer_indices],
                   showgrid=True, gridcolor="#eeeeee"),
        hovermode="closest",
        template="plotly_white",
    )

    html_filename = filename.replace(".png", ".html")
    html_fig.write_html(html_filename, include_plotlyjs="cdn")
    print(f"Saved: {html_filename}")


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
    ax.set_xticks(range(0, n_experts, 8))

    plt.tight_layout()

    if n_samples is None:
        filename = os.path.join(PLOT_DIR, f"{filename_prefix}.png")
    else:
        filename = os.path.join(PLOT_DIR, f"{filename_prefix}_n{n_samples}.png")
    plt.savefig(filename, dpi=200)
    plt.close()
    print(f"Saved: {filename}")
