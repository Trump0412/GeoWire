from __future__ import annotations

import inspect
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class QwenInspection:
    model_class: str
    candidate_modules: dict[str, str]


def inspect_qwen_modules(model: nn.Module) -> QwenInspection:
    candidates: dict[str, str] = {}
    keywords = ("visual", "vision", "merger", "projector", "language", "model")
    for name, module in model.named_modules():
        if any(k in name.lower() for k in keywords):
            try:
                sig = str(inspect.signature(module.forward))
            except (TypeError, ValueError):
                sig = "<unavailable>"
            candidates[name] = f"{module.__class__.__name__}{sig}"
    return QwenInspection(model.__class__.__name__, candidates)


class Qwen3VLGeoWireForConditionalGeneration(nn.Module):
    """Pinned-version bridge stub.

    The real bridge must copy the minimal Qwen3-VL forward path after inspection.
    This wrapper intentionally fails until the parity path is implemented for the
    installed Transformers version.
    """

    def __init__(self, base_model: nn.Module, geowire: nn.Module, token_layout_builder) -> None:
        super().__init__()
        self.base_model = base_model
        self.geowire = geowire
        self.token_layout_builder = token_layout_builder

    def forward(
        self,
        *,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.LongTensor,
        graph,
        labels: torch.LongTensor | None = None,
        **kwargs,
    ):
        raise NotImplementedError(
            "Qwen3-VL bridge must be implemented against the inspected Transformers source "
            "and pass alpha=0 parity before training."
        )
