from __future__ import annotations

import torch
from torch import nn


class DummyVisionBackend(nn.Module):
    def __init__(self, hidden_size: int = 16) -> None:
        super().__init__()
        self.proj = nn.Linear(3, hidden_size)

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        if pixels.ndim != 2 or pixels.shape[-1] != 3:
            raise ValueError("pixels must be [N, 3]")
        return self.proj(pixels)
