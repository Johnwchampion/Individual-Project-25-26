import torch
import torch.nn.functional as F


def _unpack_gate_output(outputs):
    """Extract (float_tensor, int_tensor) from gate output regardless of tuple order."""
    tensors = [t for t in outputs if torch.is_tensor(t)] if isinstance(outputs, (tuple, list)) else [outputs]
    float_t = next(t for t in tensors if t.dtype.is_floating_point)
    int_t   = next(t for t in tensors if not t.dtype.is_floating_point)
    return float_t, int_t


class ExpertSteerer:
    """
    Registers forward hooks on each targeted MoE gate to modify router logits
    before the top-k selection step.

    Hard mode:  logit[expert] = -1e9    (expert is never selected)
    Soft mode:  logit[expert] -= strength  (expert is less competitive)

    Usage:
        with ExpertSteerer(model, candidates, mode="hard") as steerer:
            output = generate(model, tokenizer, messages)
    """

    def __init__(self, model, candidates, mode="hard", strength=30.0):
        """
        candidates: {layer_index (int): [expert_indices]}
        mode:       "hard" or "soft"
        strength:   logit penalty magnitude for soft mode
        """
        self._hooks = []
        transformer = model.model

        for layer_idx, expert_indices in candidates.items():
            layer = transformer.layers[layer_idx]
            if not hasattr(layer, "mlp") or not hasattr(layer.mlp, "gate"):
                continue
            gate  = layer.mlp.gate
            top_k = getattr(gate, "top_k", 6)
            hook  = gate.register_forward_hook(
                self._make_hook(expert_indices, top_k, mode, strength)
            )
            self._hooks.append(hook)

    def _make_hook(self, expert_indices, top_k, mode, strength):
        def hook(module, inputs, outputs):
            hidden = inputs[0]
            bsz, seq_len, hidden_dim = hidden.shape
            h = hidden.reshape(-1, hidden_dim).float()

            with torch.no_grad():
                logits = F.linear(h, module.weight.float())  # [bsz*seq, n_experts]

            if mode == "hard":
                logits[:, expert_indices] = -1e9
            else:
                logits[:, expert_indices] -= strength

            scores = F.softmax(logits, dim=-1)
            topk_weights, topk_idx = torch.topk(scores, k=top_k, dim=-1, sorted=False)
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

            orig_weights, orig_idx = _unpack_gate_output(outputs)
            topk_weights = topk_weights.to(orig_weights.dtype).reshape_as(orig_weights)
            topk_idx     = topk_idx.to(orig_idx.dtype).reshape_as(orig_idx)

            return (topk_weights, topk_idx)

        return hook

    def remove(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.remove()
