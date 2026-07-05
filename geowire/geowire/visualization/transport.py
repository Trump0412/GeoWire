from __future__ import annotations

import torch


def transport_delta_norm(before: torch.Tensor, after: torch.Tensor) -> float:
    return float((after - before).norm(dim=-1).mean().detach().cpu())
