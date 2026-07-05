from __future__ import annotations

import torch

from geowire.data.tip_dataset import sample_tip_targets
from geowire.geometry.graph_builder import (
    random_same_degree_graph,
    restrict_incoming_sources,
    self_loop_graph,
    shuffled_endpoint_graph,
)
from geowire.models.geowire import GeoWireTransport
from geowire.objectives.tip import (
    direct_observation_keep_loss,
    isolation_loss,
    masked_recovery_loss,
    substitution_consistency_loss,
    total_tip_loss,
)
from geowire.types import SparseGraph


def tip_loss_step(
    model: GeoWireTransport,
    clean: torch.Tensor,
    graph: SparseGraph,
    frame_index: torch.Tensor,
    *,
    mask_ratio: float = 0.15,
    lambda_sub: float = 0.25,
    lambda_keep: float = 0.02,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute one v0.2 TIP objective on cached frozen semantic tokens."""

    targets = sample_tip_targets(graph, frame_index, mask_ratio=mask_ratio)
    mask = torch.zeros(clean.shape[0], dtype=torch.bool, device=clean.device)
    if targets.masked_node_ids.numel():
        mask[targets.masked_node_ids.to(device=clean.device)] = True
    else:
        valid_count = max(1, int(clean.shape[0] * mask_ratio))
        mask[:valid_count] = True

    hidden_masked = clean.clone()
    hidden_masked[mask] = 0.0

    pred = model(hidden_masked, graph)
    rec = masked_recovery_loss(pred, clean, mask)

    sub = rec.new_tensor(0.0)
    if targets.equivalent_support_groups:
        dst = int(targets.masked_node_ids[0])
        a, b = targets.equivalent_support_groups[0]
        graph_a = restrict_incoming_sources(graph, dst_node=dst, allowed_sources=a)
        graph_b = restrict_incoming_sources(graph, dst_node=dst, allowed_sources=b)
        pred_a = model(hidden_masked, graph_a)
        pred_b = model(hidden_masked, graph_b)
        sub_mask = torch.zeros_like(mask)
        sub_mask[dst] = True
        sub = substitution_consistency_loss(pred_a, pred_b, sub_mask)

    keep = direct_observation_keep_loss(pred, clean, ~mask)
    loss = total_tip_loss(rec=rec, sub=sub, keep=keep, lambda_sub=lambda_sub, lambda_keep=lambda_keep)
    metrics = {
        "loss": float(loss.detach()),
        "rec": float(rec.detach()),
        "sub": float(sub.detach()),
        "keep": float(keep.detach()),
        "masked_nodes": int(mask.sum().item()),
    }
    return loss, metrics


@torch.no_grad()
def graph_control_metrics(
    model: GeoWireTransport,
    clean: torch.Tensor,
    graph: SparseGraph,
    frame_index: torch.Tensor,
    *,
    mask_ratio: float = 0.15,
) -> dict[str, float]:
    """Evaluate full/self/random/shuffled graph recovery on one cached clip."""

    targets = sample_tip_targets(graph, frame_index, mask_ratio=mask_ratio)
    mask = torch.zeros(clean.shape[0], dtype=torch.bool, device=clean.device)
    if targets.masked_node_ids.numel():
        mask[targets.masked_node_ids.to(device=clean.device)] = True
    else:
        mask[: max(1, int(clean.shape[0] * mask_ratio))] = True
    hidden_masked = clean.clone()
    hidden_masked[mask] = 0.0
    full = model(hidden_masked, graph)
    self_only = model(hidden_masked, self_loop_graph(graph.num_nodes))
    random = model(hidden_masked, random_same_degree_graph(graph, seed=11))
    shuffled = model(hidden_masked, shuffled_endpoint_graph(graph, seed=17))
    full_rec = masked_recovery_loss(full, clean, mask)
    self_rec = masked_recovery_loss(self_only, clean, mask)
    random_rec = masked_recovery_loss(random, clean, mask)
    shuffled_rec = masked_recovery_loss(shuffled, clean, mask)
    return {
        "eval_full_rec": float(full_rec.detach()),
        "eval_self_loop_rec": float(self_rec.detach()),
        "eval_random_graph_rec": float(random_rec.detach()),
        "eval_shuffled_graph_rec": float(shuffled_rec.detach()),
        "eval_random_graph_gap": float((random_rec - full_rec).detach()),
        "eval_shuffled_graph_gap": float((shuffled_rec - full_rec).detach()),
    }


def run_tip_debug_step(
    hidden: torch.Tensor,
    clean: torch.Tensor,
    graph: SparseGraph,
    *,
    frame_index: torch.Tensor | None = None,
) -> dict[str, float]:
    """Run a tiny Phase 1 mechanism step without loading Qwen/VGGT.

    `hidden` stands in for masked frozen Qwen visual tokens and `clean` is the
    stop-gradient teacher. The function returns the same diagnostics used by
    real Phase 1 runs: full/self/random/shuffled graph behavior plus the v0.2
    TIP loss terms.
    """

    model = GeoWireTransport(hidden_size=hidden.shape[-1], num_blocks=2)
    for block in model.blocks:
        block.alpha.data.fill_(1.0)

    if frame_index is None:
        half = max(1, hidden.shape[0] // 2)
        frame_index = torch.cat(
            [
                torch.zeros(half, dtype=torch.long),
                torch.ones(hidden.shape[0] - half, dtype=torch.long),
            ]
        )
    loss, metrics = tip_loss_step(model, clean, graph, frame_index, mask_ratio=0.50)
    loss.backward()
    metrics.update(graph_control_metrics(model, clean, graph, frame_index, mask_ratio=0.50))
    metrics["non_edge_iso_diagnostic"] = float(isolation_loss(clean.detach(), clean.detach()).detach())
    return metrics
