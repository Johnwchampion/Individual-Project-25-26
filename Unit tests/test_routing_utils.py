from run_stage1 import slice_question_routing


def _make_trace(n_tokens, n_experts=8, top_k=6, layer="layer.0"):
    """Build a minimal routing trace."""
    flat = list(range(n_tokens * top_k))
    logits = [[float(t * n_experts + e) for e in range(n_experts)] for t in range(n_tokens)]
    return {
        "layer_traces": {
            layer: [{"top_experts": flat, "logit_scores": logits}]
        }
    }


class TestSliceQuestionRouting:

    def test_routing_flat_indexing_uses_topk(self):
        # With top_k=6, token 1 starts at index 6, not index 1.
        # A naive slice [start:end] would be wrong; correct is [start*k : end*k].
        trace = _make_trace(n_tokens=4, top_k=6)
        routing, _ = slice_question_routing(trace, start_idx=1, end_idx=3, top_k=6)
        layer = "layer.0"
        assert routing[layer] == list(range(6, 18))   # tokens 1 and 2 → indices 6..17

    def test_logits_sliced_by_token(self):
        trace = _make_trace(n_tokens=4, n_experts=8, top_k=6)
        _, logits = slice_question_routing(trace, start_idx=1, end_idx=3, top_k=6)
        layer = "layer.0"
        # Logits are indexed by token directly, not multiplied by top_k
        expected = [[float(t * 8 + e) for e in range(8)] for t in range(1, 3)]
        assert logits[layer] == expected

    def test_empty_events_layer_excluded(self):
        trace = {"layer_traces": {"layer.0": []}}
        routing, logits = slice_question_routing(trace, start_idx=0, end_idx=2)
        assert "layer.0" not in routing
        assert "layer.0" not in logits

    def test_missing_logit_scores_excluded(self):
        trace = {"layer_traces": {"layer.0": [{"top_experts": list(range(12))}]}}
        routing, logits = slice_question_routing(trace, start_idx=0, end_idx=2, top_k=6)
        assert "layer.0" in routing
        assert "layer.0" not in logits

    def test_full_span_returns_all_tokens(self):
        n_tokens = 5
        trace = _make_trace(n_tokens=n_tokens, top_k=6)
        routing, _ = slice_question_routing(trace, start_idx=0, end_idx=n_tokens, top_k=6)
        assert routing["layer.0"] == list(range(n_tokens * 6))
