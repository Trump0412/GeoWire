from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_recovery_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Cosine recovery metric/loss for masked target nodes.

    This is useful for smoke tests and diagnostics. Full Phase 1 training can
    switch to `masked_recovery_infonce_loss` to avoid average-vector collapse.
    """

    if mask.dtype != torch.bool:
        mask = mask.bool()
    if mask.sum() == 0:
        return pred.new_tensor(0.0)
    return 1.0 - F.cosine_similarity(pred[mask], target[mask].detach(), dim=-1).mean()


def masked_recovery_infonce_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    *,
    temperature: float = 0.07,
) -> torch.Tensor:
    """InfoNCE recovery loss over masked nodes against in-batch token targets."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if mask.dtype != torch.bool:
        mask = mask.bool()
    if mask.sum() == 0:
        return pred.new_tensor(0.0)
    pred_m = F.normalize(pred[mask], dim=-1)
    target_all = F.normalize(target.detach(), dim=-1)
    logits = pred_m @ target_all.T / temperature
    labels = torch.nonzero(mask, as_tuple=False).flatten().to(device=pred.device)
    return F.cross_entropy(logits, labels)


def substitution_consistency_loss(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Consistency between two masked equivalent-support transport outputs."""

    if mask is None:
        return 1.0 - F.cosine_similarity(a, b.detach(), dim=-1).mean()
    if mask.sum() == 0:
        return a.new_tensor(0.0)
    return 1.0 - F.cosine_similarity(a[mask], b[mask].detach(), dim=-1).mean()


def isolation_loss(clean: torch.Tensor, with_distractor: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """Diagnostic only: non-edge isolation is not part of v0.2 training loss."""

    return substitution_consistency_loss(clean, with_distractor, mask)


def direct_observation_keep_loss(pred: torch.Tensor, clean: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Low-weight preservation term for unmasked direct observations."""

    return masked_recovery_loss(pred, clean, mask)


def total_tip_loss(
    *,
    rec: torch.Tensor,
    sub: torch.Tensor,
    keep: torch.Tensor | None = None,
    iso: torch.Tensor | None = None,
    lambda_rec: float = 1.0,
    lambda_sub: float = 0.25,
    lambda_keep: float = 0.02,
    lambda_iso: float = 0.0,
) -> torch.Tensor:
    """Return the v0.2 TIP training objective.

    `iso` is accepted for diagnostics/backward compatibility, but its default
    coefficient is zero because non-edge isolation is structurally diagnostic in
    the frozen sparse graph setting.
    """

    loss = lambda_rec * rec + lambda_sub * sub
    if keep is not None:
        loss = loss + lambda_keep * keep
    if iso is not None and lambda_iso:
        loss = loss + lambda_iso * iso
    return loss
