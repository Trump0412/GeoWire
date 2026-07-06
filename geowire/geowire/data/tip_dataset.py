from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

from geowire.types import SparseGraph


@dataclass(frozen=True)
class TIPTargets:
    masked_node_ids: torch.Tensor
    equivalent_support_groups: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    distractor_clip_ids: tuple[str, ...] = ()
    permutation: torch.Tensor | None = None


def cross_frame_neighbor_mask(graph: SparseGraph, frame_index: torch.Tensor) -> torch.Tensor:
    """Return nodes with at least one non-self source from a different frame."""

    frame_index = frame_index.to(device=graph.dst.device)
    cross = frame_index[graph.dst] != frame_index[graph.src]
    out = torch.zeros(graph.num_nodes, dtype=torch.bool, device=graph.dst.device)
    if cross.any():
        out[graph.dst[cross]] = True
    return out.cpu()


def sample_tip_targets(
    graph: SparseGraph,
    frame_index: torch.Tensor,
    *,
    mask_ratio: float = 0.15,
    min_masked: int = 1,
    generator: torch.Generator | None = None,
) -> TIPTargets:
    """Sample Phase 1 masked targets and source substitutions from a graph."""

    if not 0 < mask_ratio <= 1:
        raise ValueError("mask_ratio must be in (0, 1]")
    eligible = torch.nonzero(cross_frame_neighbor_mask(graph, frame_index), as_tuple=False).flatten()
    if eligible.numel() == 0:
        return TIPTargets(masked_node_ids=torch.empty(0, dtype=torch.long), equivalent_support_groups=())
    frame_index_cpu = frame_index.detach().cpu()
    dst_cpu = graph.dst.detach().cpu()
    src_cpu = graph.src.detach().cpu()
    count = max(min_masked, int(round(float(eligible.numel()) * mask_ratio)))
    count = min(count, int(eligible.numel()))
    order = torch.randperm(eligible.numel(), generator=generator)
    masked = eligible[order[:count]].to(torch.long)
    groups: list[tuple[torch.Tensor, torch.Tensor]] = []
    for dst in masked.tolist():
        idx = torch.nonzero(dst_cpu == dst, as_tuple=False).flatten()
        src = src_cpu[idx]
        src = src[src != dst]
        if src.numel() < 2:
            continue
        source_frames = frame_index_cpu[src]
        unique_frames = torch.unique(source_frames)
        if unique_frames.numel() < 2:
            continue
        first_frame = unique_frames[0]
        a = src[source_frames == first_frame]
        b = src[source_frames != first_frame]
        if a.numel() and b.numel():
            groups.append((a.to(torch.long), b.to(torch.long)))
    return TIPTargets(masked_node_ids=masked, equivalent_support_groups=tuple(groups))


class TIPDataset(Dataset):
    """Dataset binding cached visual tokens, sparse graphs and sampled TIP targets."""

    def __init__(self, items: list[dict]) -> None:
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        item = dict(self.items[index])
        if "targets" not in item and "graph" in item and "frame_index" in item:
            item["targets"] = sample_tip_targets(item["graph"], item["frame_index"])
        return item
