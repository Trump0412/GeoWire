from __future__ import annotations

import torch
from torch import nn

from geowire.types import SparseGraph


def sparse_aggregate(values: torch.Tensor, graph: SparseGraph) -> torch.Tensor:
    """Aggregate values from src to dst for one flattened token sequence."""

    if values.ndim != 2:
        raise ValueError("values must be [N, D]")
    if values.shape[0] != graph.num_nodes:
        raise ValueError("values and graph node count mismatch")
    weighted = values[graph.src] * graph.weight.to(device=values.device, dtype=values.dtype).unsqueeze(-1)
    out = torch.zeros_like(values)
    out.index_add_(0, graph.dst.to(device=values.device), weighted)
    return out


class SparseTransportBlock(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.value_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.alpha = nn.Parameter(torch.zeros(()))

    def forward(self, hidden: torch.Tensor, graph: SparseGraph) -> torch.Tensor:
        values = self.value_proj(self.norm(hidden))
        transported = sparse_aggregate(values, graph)
        return hidden + self.alpha * self.out_proj(transported)
