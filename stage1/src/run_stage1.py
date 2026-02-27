import torch
import random
import config as cfg
from dataset import load_jsonl, group_into_pairs, sample_pairs, group_into_safety_pairs
from model import load_model, generate
from routing import RouterTracer
from collections import defaultdict
from visualize import (
    accumulate_expert_counts,
    compute_rd,
    rank_positive_rd,
    rank_negative_rd,
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
    Given a trace and token span [start_idx, end_idx),
    keep only routing corresponding to question tokens.
    """
    filtered = {}

    for layer_name, events in trace["layer_traces"].items():
        if not events:
            continue

        flat = events[0]["top_experts"]

        start_flat = start_idx * top_k
        end_flat = end_idx * top_k

        filtered[layer_name] = flat[start_flat:end_flat]

    return filtered

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

        question_routing = slice_question_routing(trace_with, start_idx, end_idx, top_k=6)


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

        # Slice no-context routing to QUESTION ONLY (same as with-context)
        question_routing_without = slice_question_routing(
            trace_without, start_idx_without, end_idx_without, top_k=6
        )

        accumulate_expert_counts(question_routing, count_with, tokens_with)
        accumulate_expert_counts(question_routing_without, count_without, tokens_without)

    rd_by_layer = compute_rd(
        count_with,
        tokens_with,
        count_without,
        tokens_without
    )

    print(f"Computed RD for {len(rd_by_layer)} layers")
    rank_positive_rd(rd_by_layer)
    rank_negative_rd(rd_by_layer)

    print("\nFinished all pairs")


if __name__ == "__main__":
    main()


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

    Token positions [start_idx, end_idx) map to flat indices
    [start_idx * top_k, end_idx * top_k) in the stored expert list,
    because each token contributes exactly top_k expert selections stored
    contiguously — matching the layout produced by RouterTracer.
    """
    filtered = {}
    for layer_name, events in trace["layer_traces"].items():
        if not events:
            continue

        flat = events[0]["top_experts"]
        start_flat = start_idx * top_k
        end_flat   = min(end_idx * top_k, len(flat))

        if start_flat >= end_flat:
            continue

        filtered[layer_name] = flat[start_flat:end_flat]

    return filtered


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
        routing_unsafe = slice_assistant_routing(
            trace_unsafe, unsafe_start, unsafe_start + K
        )
        routing_safe = slice_assistant_routing(
            trace_safe, safe_start, safe_start + K
        )

        accumulate_expert_counts(routing_unsafe, count_unsafe, tokens_unsafe)
        accumulate_expert_counts(routing_safe,   count_safe,   tokens_safe)

    print(f"\nSkipped {skipped} pairs (response too short, < {cfg.MIN_RESPONSE_TOKENS} tokens)")

    # RD_safety = P(expert | safe) - P(expert | unsafe)
    # Positive RD  →  expert more active during safe refusal behaviour
    # Negative RD  →  expert more active during harmful compliance
    rd_by_layer = compute_rd(
        count_safe,
        tokens_safe,
        count_unsafe,
        tokens_unsafe,
    )

    print(f"Computed RD for {len(rd_by_layer)} layers")
    rank_positive_rd(rd_by_layer)
    rank_negative_rd(rd_by_layer)

    print("\nFinished all safety pairs")
