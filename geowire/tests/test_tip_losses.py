from __future__ import annotations

import torch

from geowire.objectives.tip import (
    masked_recovery_infonce_loss,
    masked_recovery_loss,
    substitution_consistency_loss,
    total_tip_loss,
)


def test_masked_recovery_zero_for_equal_vectors() -> None:
    x = torch.randn(4, 8)
    mask = torch.tensor([True, False, True, False])
    loss = masked_recovery_loss(x, x, mask)
    assert float(loss) < 1e-6


def test_substitution_consistency() -> None:
    a = torch.tensor([[1.0, 0.0, 0.0]])
    b = torch.tensor([[0.0, 1.0, 0.0]])
    loss = substitution_consistency_loss(a, b)
    assert torch.allclose(loss, torch.tensor(1.0))


def test_total_tip_loss() -> None:
    val = total_tip_loss(
        rec=torch.tensor(1.0),
        sub=torch.tensor(2.0),
        keep=torch.tensor(3.0),
        iso=torch.tensor(99.0),
    )
    assert torch.allclose(val, torch.tensor(1.56))


def test_masked_recovery_infonce_lower_for_matching_targets() -> None:
    target = torch.eye(4)
    pred = target.clone()
    mask = torch.tensor([True, False, True, False])
    loss = masked_recovery_infonce_loss(pred, target, mask, temperature=0.07)
    assert float(loss) < 1e-3
