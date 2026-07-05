from __future__ import annotations

import torch

from geowire.geometry.graph_builder import EdgeCandidates, build_graph
from geowire.models.sparse_transport import SparseTransportBlock, sparse_aggregate


def graph():
    ones = torch.ones(2)
    return build_graph(
        3,
        [
            EdgeCandidates(
                dst=torch.tensor([1, 2]),
                src=torch.tensor([0, 1]),
                weight=ones,
                edge_type=torch.full((2,), 2, dtype=torch.uint8),
                reproj_error=torch.zeros(2),
                cycle_error=torch.zeros(2),
                visibility=ones,
                confidence=ones,
            )
        ],
    )


def test_sparse_aggregate_matches_dense() -> None:
    g = graph()
    x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    sparse = sparse_aggregate(x, g)
    dense = torch.zeros(3, 3)
    dense[g.dst, g.src] = g.weight
    expected = dense @ x
    assert torch.allclose(sparse, expected)


def test_alpha_zero_identity() -> None:
    g = graph()
    x = torch.randn(3, 8)
    block = SparseTransportBlock(8)
    y = block(x, g)
    assert torch.allclose(x, y)
