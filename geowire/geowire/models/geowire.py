from __future__ import annotations

import torch
from torch import nn

from geowire.models.sparse_transport import SparseTransportBlock
from geowire.types import SparseGraph


class GeoWireTransport(nn.Module):
    def __init__(self, hidden_size: int, num_blocks: int = 2) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([SparseTransportBlock(hidden_size) for _ in range(num_blocks)])

    def forward(self, hidden: torch.Tensor, graph: SparseGraph) -> torch.Tensor:
        for block in self.blocks:
            hidden = block(hidden, graph)
        return hidden
