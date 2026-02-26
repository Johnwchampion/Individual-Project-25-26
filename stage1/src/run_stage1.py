import torch
import random
import config as cfg
from dataset import load_jsonl, group_into_pairs, sample_pairs
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

    
