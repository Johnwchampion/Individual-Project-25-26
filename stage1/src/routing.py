from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch


@dataclass
class RouteEvent:
    top_experts: List[int]


class RouterTracer:
    """
    Records MoE routing decisions during generation.

    Hooks the router modules in DeepSeek-V2-Lite:
        model.model.layers[i].mlp.gate

    Output format (JSON-serialisable):
    {
        "example_id": str,
        "run_tag": str,
        "layer_traces": {
            "model.layers.21.mlp.gate": [ {"top_experts": [...]}, ... ],
            ...
        }
    }
    """

    def __init__(self, model: torch.nn.Module) -> None:
        self._model = model
        self._hooks: List[Any] = []
        self._router_names: List[str] = []
        self._active = False
        self._current_trace: Dict[str, Any] = {}
        self._attach_hooks()

    def start(self, example_id: str, run_tag: str) -> None:
        self._current_trace = {
            "example_id": example_id,
            "run_tag": run_tag,
            "layer_traces": {name: [] for name in self._router_names},
        }
        self._active = True

    def stop(self) -> Dict[str, Any]:
        self._active = False
        return self._current_trace

    def remove_hooks(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    # -------------------------
    # Hook attachment (clean)
    # -------------------------

    def _attach_hooks(self) -> None:
        """
        Attach forward hooks to each transformer layer's MoE router:
            layer.mlp.gate
        """
        transformer = self._get_transformer()

        if not hasattr(transformer, "layers"):
            raise RuntimeError("Unexpected model structure: transformer missing .layers")

        for layer_index, layer in enumerate(transformer.layers):
            # DeepSeek-V2-Lite MoE router lives here:
            if not hasattr(layer, "mlp") or not hasattr(layer.mlp, "gate"):
                continue

            router_module = layer.mlp.gate
            layer_name = f"model.layers.{layer_index}.mlp.gate"

            self._router_names.append(layer_name)
            hook = router_module.register_forward_hook(self._make_hook(layer_name))
            self._hooks.append(hook)

    def _get_transformer(self) -> torch.nn.Module:
        """
        HF causal LM wrappers typically expose the transformer as model.model.
        We keep this minimal but explicit.
        """
        if hasattr(self._model, "model"):
            return getattr(self._model, "model")
        raise RuntimeError("Unexpected model structure: top-level model missing .model")

    # -------------------------
    # Hook + extraction
    # -------------------------

    def _make_hook(self, layer_name: str):
        def hook(_module: torch.nn.Module, _inputs: Tuple[Any, ...], outputs: Any) -> None:
            if not self._active:
                return

            top_experts = self._extract_expert_indices(outputs)
            if top_experts is None:
                return

            event = RouteEvent(top_experts=top_experts)
            self._current_trace["layer_traces"][layer_name].append(self._event_to_dict(event))

        return hook

    def _extract_expert_indices(self, outputs: Any) -> Optional[List[int]]:
        """
        Extract expert indices from the router output.

        Current conservative strategy:
        - Look for an integer tensor in outputs (int32/int64).
        - Convert the last row (if 2D+) to a flat Python list.

        This matches what we've observed for model.layers.X.mlp.gate.
        """
        tensors: List[torch.Tensor] = []

        if isinstance(outputs, (list, tuple)):
            tensors = [x for x in outputs if torch.is_tensor(x)]
        elif torch.is_tensor(outputs):
            tensors = [outputs]

        topk_tensor: Optional[torch.Tensor] = None
        for t in tensors:
            if t.dtype in (torch.int32, torch.int64) and t.ndim >= 1:
                topk_tensor = t
                break

        return self._tensor_to_int_list(topk_tensor)

    def _tensor_to_int_list(self, tensor: Optional[torch.Tensor]) -> Optional[List[int]]:
        if tensor is None:
            return None

        data = tensor.detach().cpu()

        # Remove batch dimension only
        if data.ndim == 3:
            data = data[0]  # [seq_len, top_k]

        return [int(v) for v in data.flatten().tolist()]


    def _event_to_dict(self, event: RouteEvent) -> Dict[str, Any]:
        return {"top_experts": event.top_experts}
