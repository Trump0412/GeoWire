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


def _run_model(hidden: torch.Tensor, graph: SparseGraph) -> torch.Tensor:
    model = GeoWireTransport(hidden_size=hidden.shape[-1], num_blocks=2)
    for block in model.blocks:
        block.alpha.data.fill_(1.0)
    return model(hidden, graph)


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
    targets = sample_tip_targets(graph, frame_index, mask_ratio=0.50)
    mask = torch.zeros(hidden.shape[0], dtype=torch.bool, device=hidden.device)
    if targets.masked_node_ids.numel():
        mask[targets.masked_node_ids.to(device=hidden.device)] = True
    else:
        mask[::2] = True

    hidden_masked = hidden.clone()
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

    keep_mask = ~mask
    keep = direct_observation_keep_loss(pred, clean, keep_mask)
    iso = isolation_loss(pred.detach(), pred.detach(), mask)
    loss = total_tip_loss(rec=rec, sub=sub, keep=keep)
    loss.backward()

    with torch.no_grad():
        self_pred = model(hidden_masked, self_loop_graph(graph.num_nodes))
        random_pred = model(hidden_masked, random_same_degree_graph(graph, seed=11))
        shuffled_pred = model(hidden_masked, shuffled_endpoint_graph(graph, seed=17))
        self_rec = masked_recovery_loss(self_pred, clean, mask)
        random_rec = masked_recovery_loss(random_pred, clean, mask)
        shuffled_rec = masked_recovery_loss(shuffled_pred, clean, mask)

    return {
        "loss": float(loss.detach()),
        "rec": float(rec.detach()),
        "sub": float(sub.detach()),
        "keep": float(keep.detach()),
        "non_edge_iso_diagnostic": float(iso.detach()),
        "self_loop_rec": float(self_rec.detach()),
        "random_graph_rec": float(random_rec.detach()),
        "shuffled_graph_rec": float(shuffled_rec.detach()),
        "random_graph_gap": float((random_rec - rec).detach()),
        "shuffled_graph_gap": float((shuffled_rec - rec).detach()),
        "masked_nodes": int(mask.sum().item()),
    }
