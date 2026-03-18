import torch
import random
import argparse
import config as cfg
from dataset import load_jsonl, group_into_pairs, sample_pairs, group_into_safety_pairs
from model import load_model, generate
from routing import RouterTracer
from collections import defaultdict
from visualize import (
    accumulate_expert_counts,
    accumulate_mean_logits,
    compute_layer_token_differences,
    compute_rd,
    compute_rd_logits,
    plot_layer_changes,
    plot_rd_scatter,
    rank_positive_rd,
    rank_negative_rd,
    save_rd,
)


def extract_question(messages):
    for msg in messages:
        if msg["role"] == "user":
            content = msg["content"]

            if "Question:" in content:
                return content.split("Question:")[-1].strip()

            return content.strip()  # fallback

    raise ValueError("No user message found")

def find_subsequence(sequence, subseq):

    for i in range(len(sequence) - len(subseq) + 1):
        if sequence[i:i + len(subseq)] == subseq:
            return i
    return None

def slice_question_routing(trace, start_idx, end_idx, top_k=6):
    """
    Given a trace and token span [start_idx, end_idx), extract routing for
    those tokens only.

    Returns a tuple (routing, logits):
      routing : {layer_name: flat list of top-k indices for the token window}
      logits  : {layer_name: [[float * n_experts] * n_tokens] for the window}
    """
    routing = {}
    logits = {}

    for layer_name, events in trace["layer_traces"].items():
        if not events:
            continue

        event = events[0]
        flat = event["top_experts"]
        routing[layer_name] = flat[start_idx * top_k : end_idx * top_k]

        if event.get("logit_scores"):
            logits[layer_name] = event["logit_scores"][start_idx:end_idx]

    return routing, logits

def main():
    #Load data
    records = load_jsonl(cfg.DATA_DIR)
    pairs = group_into_pairs(records)
    random.shuffle(pairs)
    sampled_pairs = pairs[:500]  # Adjust as needed for testing or full run

    print(f"Sampled pairs: {len(sampled_pairs)}")


    #Load model
    model, tokenizer = load_model(cfg.MODEL_NAME, cfg.CACHE_DIR)
    model.eval()

    #Prepare tracing
    tracer = RouterTracer(model)

    count_with = {}
    count_without = {}
    tokens_with = defaultdict(int)
    tokens_without = defaultdict(int)
    logit_sum_with = {}
    logit_sum_without = {}
    logit_tokens_with = defaultdict(int)
    logit_tokens_without = defaultdict(int)
    diff_sum_by_layer = defaultdict(int)
    diff_tokens_by_layer = defaultdict(int)

    # Iterate over pairs
    for idx, pair in enumerate(sampled_pairs, start=1):
        if idx == 1 or idx % 50 == 0:
            print(f"Processing pair {idx}/{len(sampled_pairs)}")

        #With context
        inputs_with = tokenizer.apply_chat_template(
            pair.with_context.messages,
            return_tensors="pt",
            add_generation_prompt=False
        ).to(model.device)

        tracer.start(example_id=pair.base_id, run_tag="with_context")

        with torch.no_grad():
            model(inputs_with)

        trace_with = tracer.stop()

        #Identify question span
        question = extract_question(pair.with_context.messages)
        full_ids = inputs_with[0].tolist()
        question_ids = tokenizer(
            question,
            add_special_tokens=False
        )["input_ids"]

        start_idx = find_subsequence(full_ids, question_ids)

        if start_idx is None:
            print("Question tokens not found in input sequence")
            continue

        end_idx = start_idx + len(question_ids)

        question_routing, question_logits = slice_question_routing(trace_with, start_idx, end_idx, top_k=6)

        #Without context
        inputs_without = tokenizer.apply_chat_template(
            pair.no_context.messages,
            return_tensors="pt",
            add_generation_prompt=False
        ).to(model.device)

        tracer.start(example_id=pair.base_id, run_tag="no_context")

        with torch.no_grad():
            model(inputs_without)

        trace_without = tracer.stop()

        #Identify question span
        full_ids_without = inputs_without[0].tolist()
        start_idx_without = find_subsequence(full_ids_without, question_ids)

        if start_idx_without is None:
            print("Question tokens not found in NO-context input sequence")
            continue

        end_idx_without = start_idx_without + len(question_ids)

        question_routing_without, question_logits_without = slice_question_routing(
            trace_without, start_idx_without, end_idx_without, top_k=6
        )

        diff_counts = compute_layer_token_differences(
            question_routing,
            question_routing_without,
            return_counts=True,
        )

        for layer, (diff_sum, token_count) in diff_counts.items():
            diff_sum_by_layer[layer] += int(diff_sum)
            diff_tokens_by_layer[layer] += int(token_count)

        accumulate_expert_counts(question_routing, count_with, tokens_with)
        accumulate_expert_counts(question_routing_without, count_without, tokens_without)
        accumulate_mean_logits(question_logits, logit_sum_with, logit_tokens_with)
        accumulate_mean_logits(question_logits_without, logit_sum_without, logit_tokens_without)

    rd_by_layer = compute_rd(
        count_with, tokens_with,
        count_without, tokens_without,
    )
    rd_logits_by_layer = compute_rd_logits(
        logit_sum_with, logit_tokens_with,
        logit_sum_without, logit_tokens_without,
    )

    save_rd(rd_by_layer,        cfg.RD_FAITH_PATH)
    save_rd(rd_logits_by_layer, cfg.RD_FAITH_LOGITS_PATH)

    print(f"Computed RD for {len(rd_by_layer)} layers")
    layer_means = {
        layer: (diff_sum_by_layer[layer] / diff_tokens_by_layer[layer])
        for layer in diff_sum_by_layer
        if diff_tokens_by_layer[layer] > 0
    }
    if layer_means:
        plot_layer_changes(
            layer_means,
            n_samples=len(sampled_pairs),
            filename_prefix="routing_instability_faithfulness",
            title="Routing Instability: With Context vs No Context",
        )
    plot_rd_scatter(
        rd_by_layer,
        n_samples=len(sampled_pairs),
        filename_prefix="rd_scatter_faithfulness",
        title="Expert RD Scatter: With Context vs No Context",
        label_a="With Context", label_b="No Context",
        x_lim=(-1, 1),
    )
    plot_rd_scatter(
        rd_logits_by_layer,
        n_samples=len(sampled_pairs),
        filename_prefix="rd_scatter_faithfulness_logits",
        title="Expert Logit RD Scatter: With Context vs No Context",
        label_a="With Context", label_b="No Context",
        log_scale=True,
    )
    rank_positive_rd(rd_by_layer, filename_prefix="rd_rank_faith_positive")
    rank_negative_rd(rd_by_layer, filename_prefix="rd_rank_faith_negative")

    print("\nFinished all pairs")


# ===========================================================================
# Safety pipeline
# ===========================================================================

def find_assistant_start(tokenizer, messages):
    """
    Return the token index where the assistant response begins in the full
    conversation sequence.

    Strategy: tokenize all turns except the final assistant turn with
    add_generation_prompt=True. The length of this prefix equals the start
    index of the assistant response in the full sequence — analogous to how
    find_subsequence locates question tokens in the faithfulness pipeline.
    """
    without_assistant = messages[:-1]  # drop the last (assistant) turn
    prefix_ids = tokenizer.apply_chat_template(
        without_assistant,
        add_generation_prompt=True,
        return_tensors=None,          # plain Python list, no batch dim
    )
    return len(prefix_ids)


def slice_assistant_routing(trace, start_idx, end_idx, top_k=6):
    """
    Slice a full-sequence routing trace to cover only assistant response tokens.

    Returns a tuple (routing, logits):
      routing : {layer_name: flat list of top-k indices for the token window}
      logits  : {layer_name: [[float * n_experts] * n_tokens] for the window}
    """
    routing = {}
    logits = {}

    for layer_name, events in trace["layer_traces"].items():
        if not events:
            continue

        event = events[0]
        flat = event["top_experts"]
        start_flat = start_idx * top_k
        end_flat   = min(end_idx * top_k, len(flat))

        if start_flat >= end_flat:
            continue

        routing[layer_name] = flat[start_flat:end_flat]

        if event.get("logit_scores"):
            logit_end = min(end_idx, len(event["logit_scores"]))
            logits[layer_name] = event["logit_scores"][start_idx:logit_end]

    return routing, logits


def run_safety():
    # Load and shuffle all safety pairs from the BeaverTails JSONL
    records = load_jsonl(cfg.SAFETY_DATA_DIR)
    pairs   = group_into_safety_pairs(records)
    random.shuffle(pairs)

    print(f"Total safety pairs loaded: {len(pairs)}")

    # Load the Chat model — routing is measured over safety-trained weights,
    # which is where alignment behaviour actually lives.
    model, tokenizer = load_model(cfg.SAFETY_MODEL_NAME, cfg.CACHE_DIR)
    model.eval()

    # Attach forward hooks to all MoE gate modules in the model
    tracer = RouterTracer(model)

    # Per-expert activation counts and total processed token counts,
    # accumulated separately for each condition across all pairs.
    count_safe    = {}
    count_unsafe  = {}
    tokens_safe   = defaultdict(int)
    tokens_unsafe = defaultdict(int)
    logit_sum_safe    = {}
    logit_sum_unsafe  = {}
    logit_tokens_safe   = defaultdict(int)
    logit_tokens_unsafe = defaultdict(int)

    # Per-token instability accumulators — kept to demonstrate that naive
    # position-by-position comparison is invalid for safety (different tokens
    # at each position trivially produce near-maximum instability).
    diff_sum_by_layer    = defaultdict(int)
    diff_tokens_by_layer = defaultdict(int)

    skipped = 0

    for idx, pair in enumerate(pairs, start=1):
        if idx == 1 or idx % 50 == 0:
            print(f"Processing pair {idx}/{len(pairs)}")

        # Tokenize both conditions upfront (cheap CPU work) so we can check
        # response length and skip short pairs before any GPU forward passes.
        inputs_unsafe = tokenizer.apply_chat_template(
            pair.unsafe.messages,
            return_tensors="pt",
            add_generation_prompt=False,
        ).to(model.device)

        inputs_safe = tokenizer.apply_chat_template(
            pair.safe.messages,
            return_tensors="pt",
            add_generation_prompt=False,
        ).to(model.device)

        # Locate where the assistant response begins in each token sequence
        unsafe_start = find_assistant_start(tokenizer, pair.unsafe.messages)
        safe_start   = find_assistant_start(tokenizer, pair.safe.messages)

        unsafe_end = inputs_unsafe.shape[-1]
        safe_end   = inputs_safe.shape[-1]

        # Length-match: use K = min(L_safe, L_unsafe) response tokens so that
        # neither condition contributes more routing signal than the other —
        # prevents longer refusals from dominating the expert counts.
        L_unsafe = unsafe_end - unsafe_start
        L_safe   = safe_end   - safe_start
        K = min(L_unsafe, L_safe)

        if K < cfg.MIN_RESPONSE_TOKENS:
            skipped += 1
            continue

        # Unsafe pass: teacher-forced forward over [user + harmful response]
        tracer.start(example_id=pair.base_id, run_tag="unsafe")
        with torch.no_grad():
            model(inputs_unsafe)
        trace_unsafe = tracer.stop()

        # Safe pass: teacher-forced forward over [user + safe refusal]
        tracer.start(example_id=pair.base_id, run_tag="safe")
        with torch.no_grad():
            model(inputs_safe)
        trace_safe = tracer.stop()

        # Slice routing to assistant response tokens only (length-matched to K)
        routing_unsafe, logits_unsafe = slice_assistant_routing(
            trace_unsafe, unsafe_start, unsafe_start + K
        )
        routing_safe, logits_safe = slice_assistant_routing(
            trace_safe, safe_start, safe_start + K
        )

        accumulate_expert_counts(routing_unsafe, count_unsafe, tokens_unsafe)
        accumulate_expert_counts(routing_safe,   count_safe,   tokens_safe)
        accumulate_mean_logits(logits_unsafe, logit_sum_unsafe, logit_tokens_unsafe)
        accumulate_mean_logits(logits_safe,   logit_sum_safe,   logit_tokens_safe)

        # Naive per-token instability (invalid for safety — included only to
        # demonstrate the methodological flaw: different tokens at each position
        # cause trivially near-maximum instability regardless of routing).
        diff_counts = compute_layer_token_differences(
            routing_safe, routing_unsafe, return_counts=True
        )
        for layer, (diff_sum, token_count) in diff_counts.items():
            diff_sum_by_layer[layer]    += int(diff_sum)
            diff_tokens_by_layer[layer] += int(token_count)

    print(f"\nSkipped {skipped} pairs (response too short, < {cfg.MIN_RESPONSE_TOKENS} tokens)")

    # RD_safety = P(expert | safe) - P(expert | unsafe)
    # Positive RD  →  expert more active during safe refusal behaviour
    # Negative RD  →  expert more active during harmful compliance
    rd_by_layer = compute_rd(
        count_safe, tokens_safe,
        count_unsafe, tokens_unsafe,
    )
    rd_logits_by_layer = compute_rd_logits(
        logit_sum_safe, logit_tokens_safe,
        logit_sum_unsafe, logit_tokens_unsafe,
    )

    save_rd(rd_by_layer,        cfg.RD_SAFETY_PATH)
    save_rd(rd_logits_by_layer, cfg.RD_SAFETY_LOGITS_PATH)

    print(f"Computed RD for {len(rd_by_layer)} layers")

    n_processed = len(pairs) - skipped
    layer_means = {
        layer: (diff_sum_by_layer[layer] / diff_tokens_by_layer[layer])
        for layer in diff_sum_by_layer
        if diff_tokens_by_layer[layer] > 0
    }
    if layer_means:
        plot_layer_changes(
            layer_means,
            n_samples=n_processed,
            filename_prefix="routing_instability_safety_naive",
            title="Routing Instability (Safety): Naive Per-Token Comparison — Invalid Baseline",
        )
    if rd_by_layer:
        plot_rd_scatter(
            rd_by_layer,
            n_samples=n_processed,
            filename_prefix="rd_scatter_safety",
            title="Expert RD Scatter: Safe Refusal vs Unsafe Compliance",
            label_a="Safe Refusal", label_b="Unsafe Compliance",
            x_lim=(-1, 1),
        )
    if rd_logits_by_layer:
        plot_rd_scatter(
            rd_logits_by_layer,
            n_samples=n_processed,
            filename_prefix="rd_scatter_safety_logits",
            title="Expert Logit RD Scatter: Safe Refusal vs Unsafe Compliance",
            label_a="Safe Refusal", label_b="Unsafe Compliance",
            log_scale=True,
        )
    rank_positive_rd(rd_by_layer, filename_prefix="rd_rank_safety_positive")
    rank_negative_rd(rd_by_layer, filename_prefix="rd_rank_safety_negative")

    print("\nFinished all safety pairs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["faith", "safety", "both"],
        default="both",
        help="Which pipeline to run",
    )
    args = parser.parse_args()

    if args.mode == "faith":
        main()
    elif args.mode == "safety":
        run_safety()
    else:
        main()
        run_safety()
