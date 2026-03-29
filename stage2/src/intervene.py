import torch
import torch.nn.functional as F


class ExpertSteerer:
    """
    Attaches inference-time hooks to DeepSeek-V2's MoE gates to suppress
    or reweight targeted experts.

    Hard mode  (SteerMoE-faithful pre-selection):
        A forward pre-hook modifies the gate's hidden-state input so that
        suppressed experts' logits are driven to TARGET = -1e4 before
        DeepSeek's native grouped top-k runs.  δh is computed per token
        via the pseudoinverse: δh = δ_logit @ (WWᵀ)⁻¹ @ W, where
        δ_logit[ei] = TARGET − current_logit[ei].  Non-suppressed logits
        are unchanged exactly.  The gate handles all routing natively —
        grouped top-k, aux loss, load balancing — preserving the model's
        structural assumptions.

    Soft mode  (novel continuous perturbation):
        Precomputes a constant δh = δ_logit @ (WWᵀ)⁻¹ @ W where
        δ_logit[ei] = strength × RD_score[ei].  Adding this to the gate
        input shifts all targeted experts' logits by a fixed amount; the
        gate routes normally on the perturbed distribution.  Unlike hard
        mode, experts are not guaranteed to be displaced — whether they
        fall out of top-k depends on their routing margin vs the shift.

    token_range: optional (start, end) tuple. When given, hooks only modify
        positions [start:end] in the sequence dimension, matching the
        question-span used during Stage 1 RD measurement.

    Call steerer.remove() when done, or use as a context manager.
    """

    def __init__(self, model, candidates, mode="hard", strength=1.0, token_range=None):
        """
        candidates:   {layer_index (int): [expert_indices]}        for hard mode
                      {layer_index (int): {expert_idx: rd_score}} for soft mode
        mode:         "hard" or "soft"
        strength:     scaling factor applied to RD scores (soft mode only)
        token_range:  optional (start, end) ints; None = all tokens
        """
        self._hooks = []
        transformer = model.model

        for layer_idx, expert_data in candidates.items():
            layer = transformer.layers[layer_idx]
            if not hasattr(layer, "mlp") or not hasattr(layer.mlp, "gate"):
                continue
            gate = layer.mlp.gate

            if mode == "hard":
                precomp = self._compute_hard_precomp(gate, set(expert_data))
                hook = gate.register_forward_pre_hook(
                    self._make_hard_pre_hook(precomp, token_range)
                )
                self._hooks.append(hook)
            else:
                delta_h = self._compute_delta_h(gate, expert_data, strength)
                hook = gate.register_forward_pre_hook(
                    self._make_soft_pre_hook(delta_h, token_range)
                )
                self._hooks.append(hook)

    # ------------------------------------------------------------------
    # Hard mode — SteerMoE-faithful pre-selection via pseudoinverse
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_hard_precomp(gate, suppressed_set):
        """
        Precompute the rows of P = (WWᵀ)⁻¹ @ W and W for suppressed experts.

        Per-token δh = δ_logit_sup @ P_rows drives suppressed logits to TARGET
        and leaves all other logits exactly unchanged (proved by P @ Wᵀ = I).
        """
        W = gate.weight.data.float()          # [n_experts, d_model]
        WWT_inv = torch.linalg.inv(W @ W.T)   # [n_experts, n_experts]
        P = WWT_inv @ W                        # [n_experts, d_model]
        idx = sorted(suppressed_set)
        P_rows = P[idx].to(gate.weight.dtype)  # [n_suppressed, d_model]
        W_rows = W[idx].to(gate.weight.dtype)  # [n_suppressed, d_model]
        return P_rows, W_rows

    @staticmethod
    def _make_hard_pre_hook(precomp, token_range=None):
        """
        Forward pre-hook implementing SteerMoE-style pre-selection.

        Computes per-token δh such that F.linear(h + δh, W)[ei] = TARGET
        for each suppressed expert ei.  The gate then runs its complete
        native routing (grouped top-k, aux loss) on the modified input.
        """
        TARGET = -1e4
        P_rows, W_rows = precomp  # [n_suppressed, d_model]

        def hook(module, args):
            h = args[0]                       # [bsz, seq_len, d_model]
            bsz, seq_len, d = h.shape
            P = P_rows.float()
            W = W_rows.float()

            if token_range is not None:
                start, end = token_range
                end = min(end, seq_len)
                if start >= end:
                    return
                h_slice = h[:, start:end, :].float().reshape(-1, d)  # [n_tok, d]
                logits_sup  = h_slice @ W.T                           # [n_tok, n_sup]
                delta_logit = TARGET - logits_sup                     # [n_tok, n_sup]
                delta_h     = (delta_logit @ P).reshape(bsz, end - start, d)
                h_mod = h.clone()
                h_mod[:, start:end, :] = h_mod[:, start:end, :] + delta_h.to(h.dtype)
                return (h_mod,) + args[1:]

            h_flat      = h.float().reshape(-1, d)
            logits_sup  = h_flat @ W.T                                # [bsz*seq, n_sup]
            delta_logit = TARGET - logits_sup                         # [bsz*seq, n_sup]
            delta_h     = (delta_logit @ P).reshape_as(h)
            return (h + delta_h.to(h.dtype),) + args[1:]

        return hook

    # ------------------------------------------------------------------
    # Soft mode — constant pseudoinverse perturbation (novel method)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_delta_h(gate, rd_scores, strength):
        """
        Precompute constant h perturbation for soft mode.

        δh = δ_logit @ (WWᵀ)⁻¹ @ W  where  δ_logit[ei] = strength × RD[ei].
        Same δh is added to every token's hidden state — input-independent.
        """
        W = gate.weight.data.float()
        n_experts = W.shape[0]
        delta_logit = torch.zeros(n_experts, dtype=torch.float32, device=W.device)
        for ei, rd in rd_scores.items():
            delta_logit[ei] = strength * rd
        WWT_inv = torch.linalg.inv(W @ W.T)
        delta_h = delta_logit @ WWT_inv @ W    # [d_model]
        return delta_h.to(gate.weight.dtype)

    @staticmethod
    def _make_soft_pre_hook(delta_h, token_range=None):
        def hook(module, args):
            h = args[0]
            if token_range is not None:
                start, end = token_range
                h = h.clone()
                h[:, start:end, :] = h[:, start:end, :] + delta_h
                return (h,) + args[1:]
            return (h + delta_h,) + args[1:]
        return hook

    def remove(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.remove()
