from __future__ import annotations

import torch

from geowire.geometry.certify_edges import certify_edges, geometric_weight


def test_certify_edges_thresholds() -> None:
    mask = certify_edges(
        visibility=torch.tensor([0.9, 0.1, 0.9]),
        confidence=torch.tensor([0.9, 0.9, 0.1]),
        reproj_error=torch.tensor([1.0, 1.0, 1.0]),
    )
    assert mask.tolist() == [True, False, False]


def test_geometric_weight_positive() -> None:
    weight = geometric_weight(
        visibility=torch.tensor([1.0]),
        confidence=torch.tensor([0.5]),
        reproj_error=torch.tensor([0.0]),
    )
    assert torch.allclose(weight, torch.tensor([0.5]))
