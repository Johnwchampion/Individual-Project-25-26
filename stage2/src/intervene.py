import torch
import torch.nn.functional as F


class ExpertSteerer:
    """
    Expert steering for DeepSeek-V2's MoE gates.

    Hard mode (candidates: {layer: [expert_indices]}):
        Two-hook architecture. A pre-hook caches the gate's log-softmax routing
        scores. An output hook intercepts (topk_idx, topk_weight, aux_loss) and
        replaces any suppressed expert that was selected with the
        highest-scoring non-suppressed alternative from the cached scores.
        Weights are recomputed via softmax over the replacement experts' scores.
        Exactly k experts always contribute. Only tokens that had a suppressed
        expert selected are modified; all other tokens are untouched.

    Soft mode (candidates: {layer: {expert_idx: rd_score}}):
        Pre-selection only. The desired logit shift δ_logit[i] = strength * Δi
        is injected by perturbing the gate's input hidden state h before the
        gate runs. The perturbation δh is precomputed as the minimum-norm
        solution to F.linear(h + δh, W) = F.linear(h, W) + δ_logit, which is:

            δh = δ_logit @ (W Wᵀ)⁻¹ @ W

        The gate's grouped top-k then runs normally on the shifted logits.
        Always k experts with natural, in-distribution weights. Positive
        strength boosts high-RD (safe/faithful) experts; negative reverses.

    Usage:
        steerer = ExpertSteerer(model, candidates, mode="hard")
        output  = generate(model, tokenizer, messages)
        steerer.remove()

        # or as a context manager:
        with ExpertSteerer(model, candidates, mode="hard"):
            output = generate(model, tokenizer, messages)
    """

    def __init__(self, model, candidates, mode="hard", strength=1.0):
        """
        candidates: {layer_index (int): [expert_indices]}        for hard mode
                    {layer_index (int): {expert_idx: rd_score}} for soft mode
        mode:       "hard" or "soft"
        strength:   scaling factor applied to RD scores (default 1.0)
                    use negative values to reverse steering direction
        """
        self._hooks = []
        transformer = model.model

        for layer_idx, expert_data in candidates.items():
            layer = transformer.layers[layer_idx]
            if not hasattr(layer, "mlp") or not hasattr(layer.mlp, "gate"):
                continue
            gate = layer.mlp.gate

            if mode == "hard":
                pre_hook = gate.register_forward_pre_hook(self._make_pre_hook())
                self._hooks.append(pre_hook)
                out_hook = gate.register_forward_hook(
                    self._make_hard_hook(set(expert_data))
                )
                self._hooks.append(out_hook)
            else:
                delta_h = self._compute_delta_h(gate, expert_data, strength)
                soft_hook = gate.register_forward_pre_hook(
                    self._make_soft_pre_hook(delta_h)
                )
                self._hooks.append(soft_hook)

    @staticmethod
    def _compute_delta_h(gate, rd_scores, strength):
        """
        Precompute h perturbation for soft pre-selection.

        We want F.linear(h + δh, W) = F.linear(h, W) + δ_logit, i.e.
        δh @ Wᵀ = δ_logit. Minimum-norm solution (W is fat: n_experts < d_model):
            δh = δ_logit @ (W Wᵀ)⁻¹ @ W
        """
        W = gate.weight.data.float()          # [n_experts, d_model]
        n_experts = W.shape[0]

        delta_logit = torch.zeros(n_experts, dtype=torch.float32, device=W.device)
        for ei, rd in rd_scores.items():
            delta_logit[ei] = strength * rd

        WWT = W @ W.T                          # [n_experts, n_experts]
        WWT_inv = torch.linalg.inv(WWT)        # [n_experts, n_experts]
        delta_h = delta_logit @ WWT_inv @ W    # [d_model]
        return delta_h.to(gate.weight.dtype)

    @staticmethod
    def _make_soft_pre_hook(delta_h):
        def hook(module, args):
            h = args[0]
            return (h + delta_h,) + args[1:]
        return hook

    @staticmethod
    def _make_pre_hook():
        def hook(module, args):
            h = args[0]
            h_flat = h.reshape(-1, h.shape[-1])
            logits = F.linear(h_flat, module.weight)
            module._cached_log_scores = F.log_softmax(logits, dim=-1)
        return hook

    @staticmethod
    def _make_hard_hook(suppressed_set):
        def hook(module, inputs, outputs):
            topk_idx    = outputs[0].clone()
            topk_weight = outputs[1].clone()   # start from gate's original weights
            aux_loss    = outputs[2] if len(outputs) > 2 else None
            log_scores  = module._cached_log_scores  # [n_tokens, n_experts]

            # Mask suppressed experts to -inf so they can't be chosen
            mod_scores = log_scores.clone()
            for ei in suppressed_set:
                mod_scores[:, ei] = float("-inf")

            # Find tokens where a suppressed expert was selected
            suppressed_t = torch.tensor(
                sorted(suppressed_set), device=topk_idx.device
            )
            needs = (topk_idx.unsqueeze(-1) == suppressed_t).any(dim=-1).any(dim=-1)

            if needs.any():
                k = topk_idx.shape[-1]
                _, new_idx = mod_scores[needs].topk(k, dim=-1)
                topk_idx[needs] = new_idx
                # Only recompute weights for the tokens that needed replacement
                selected_scores = mod_scores[needs].gather(1, new_idx)
                topk_weight[needs] = F.softmax(selected_scores, dim=-1).to(topk_weight.dtype)

            # Tokens that didn't need replacement keep the gate's original weights
            return (topk_idx, topk_weight, aux_loss)

        return hook

    def remove(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.remove()
