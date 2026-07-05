from __future__ import annotations

import torch


def certify_edges(
    *,
    visibility: torch.Tensor,
    confidence: torch.Tensor,
    reproj_error: torch.Tensor,
    cycle_error: torch.Tensor | None = None,
    visibility_min: float = 0.5,
    confidence_min: float = 0.5,
    reproj_error_px_max: float = 8.0,
    cycle_error_px_max: float = 6.0,
) -> torch.Tensor:
    """Return a boolean mask for candidate geometric edges."""

    mask = (
        torch.isfinite(reproj_error)
        & (visibility >= visibility_min)
        & (confidence >= confidence_min)
        & (reproj_error <= reproj_error_px_max)
    )
    if cycle_error is not None:
        mask = mask & torch.isfinite(cycle_error) & (cycle_error <= cycle_error_px_max)
    return mask


def geometric_weight(
    *,
    visibility: torch.Tensor,
    confidence: torch.Tensor,
    reproj_error: torch.Tensor,
    cycle_error: torch.Tensor | None = None,
    tau_reproj: float = 4.0,
    tau_cycle: float = 4.0,
) -> torch.Tensor:
    weight = visibility * confidence * torch.exp(-reproj_error.clamp_min(0) / tau_reproj)
    if cycle_error is not None:
        weight = weight * torch.exp(-cycle_error.clamp_min(0) / tau_cycle)
    return weight.clamp_min(0)
