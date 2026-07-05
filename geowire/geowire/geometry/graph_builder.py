from __future__ import annotations

from dataclasses import dataclass

import torch

from geowire.constants import EDGE_SELF
from geowire.types import SparseGraph


@dataclass(frozen=True)
class EdgeCandidates:
    dst: torch.Tensor
    src: torch.Tensor
    weight: torch.Tensor
    edge_type: torch.Tensor
    reproj_error: torch.Tensor
    cycle_error: torch.Tensor
    visibility: torch.Tensor
    confidence: torch.Tensor


def self_loop_candidates(num_nodes: int, weight: float = 1.0) -> EdgeCandidates:
    idx = torch.arange(num_nodes, dtype=torch.long)
    nan = torch.full((num_nodes,), float("nan"), dtype=torch.float32)
    ones = torch.ones((num_nodes,), dtype=torch.float32)
    return EdgeCandidates(
        dst=idx,
        src=idx,
        weight=torch.full((num_nodes,), float(weight), dtype=torch.float32),
        edge_type=torch.full((num_nodes,), EDGE_SELF, dtype=torch.uint8),
        reproj_error=nan.clone(),
        cycle_error=nan.clone(),
        visibility=ones.clone(),
        confidence=ones.clone(),
    )


def concat_candidates(candidates: list[EdgeCandidates]) -> EdgeCandidates:
    if not candidates:
        empty_long = torch.empty(0, dtype=torch.long)
        empty_float = torch.empty(0, dtype=torch.float32)
        return EdgeCandidates(
            empty_long,
            empty_long,
            empty_float,
            torch.empty(0, dtype=torch.uint8),
            empty_float,
            empty_float,
            empty_float,
            empty_float,
        )
    return EdgeCandidates(
        dst=torch.cat([c.dst.to(torch.long) for c in candidates]),
        src=torch.cat([c.src.to(torch.long) for c in candidates]),
        weight=torch.cat([c.weight.to(torch.float32) for c in candidates]),
        edge_type=torch.cat([c.edge_type.to(torch.uint8) for c in candidates]),
        reproj_error=torch.cat([c.reproj_error.to(torch.float32) for c in candidates]),
        cycle_error=torch.cat([c.cycle_error.to(torch.float32) for c in candidates]),
        visibility=torch.cat([c.visibility.to(torch.float32) for c in candidates]),
        confidence=torch.cat([c.confidence.to(torch.float32) for c in candidates]),
    )


def topk_cross_frame(
    candidates: EdgeCandidates,
    frame_index: torch.Tensor,
    topk: int,
) -> EdgeCandidates:
    """Keep at most top-k cross-frame non-self incoming edges per destination."""

    if topk <= 0 or candidates.dst.numel() == 0:
        return candidates
    keep = torch.ones(candidates.dst.numel(), dtype=torch.bool)
    dst_frames = frame_index[candidates.dst]
    src_frames = frame_index[candidates.src]
    cross = dst_frames != src_frames
    for dst in torch.unique(candidates.dst[cross]).tolist():
        idx = torch.nonzero((candidates.dst == dst) & cross, as_tuple=False).flatten()
        if idx.numel() > topk:
            order = torch.argsort(candidates.weight[idx], descending=True)
            drop = idx[order[topk:]]
            keep[drop] = False
    return _mask_candidates(candidates, keep)


def merge_and_normalize(num_nodes: int, candidates: EdgeCandidates) -> SparseGraph:
    """Merge duplicate (dst, src) edges and row-normalize by destination."""

    if candidates.dst.numel() == 0:
        raise ValueError("Sparse graph needs at least one edge")

    merged: dict[tuple[int, int], dict[str, float | int]] = {}
    for i in range(candidates.dst.numel()):
        dst = int(candidates.dst[i])
        src = int(candidates.src[i])
        if dst < 0 or dst >= num_nodes or src < 0 or src >= num_nodes:
            raise ValueError("Edge index out of bounds")
        key = (dst, src)
        item = merged.setdefault(
            key,
            {
                "weight": 0.0,
                "edge_type": 0,
                "reproj_error": float("nan"),
                "cycle_error": float("nan"),
                "visibility": 0.0,
                "confidence": 0.0,
            },
        )
        item["weight"] = float(item["weight"]) + float(candidates.weight[i])
        item["edge_type"] = int(item["edge_type"]) | int(candidates.edge_type[i])
        item["reproj_error"] = _nanmin(float(item["reproj_error"]), float(candidates.reproj_error[i]))
        item["cycle_error"] = _nanmin(float(item["cycle_error"]), float(candidates.cycle_error[i]))
        item["visibility"] = max(float(item["visibility"]), float(candidates.visibility[i]))
        item["confidence"] = max(float(item["confidence"]), float(candidates.confidence[i]))

    keys = sorted(merged)
    dst = torch.tensor([k[0] for k in keys], dtype=torch.long)
    src = torch.tensor([k[1] for k in keys], dtype=torch.long)
    weight = torch.tensor([merged[k]["weight"] for k in keys], dtype=torch.float32)
    if (weight < 0).any() or not torch.isfinite(weight).all():
        raise ValueError("Edge weights must be finite and non-negative")
    denom = torch.zeros(num_nodes, dtype=torch.float32)
    denom.index_add_(0, dst, weight)
    if (denom == 0).any():
        missing = torch.nonzero(denom == 0, as_tuple=False).flatten().tolist()
        raise ValueError(f"Nodes without incoming edges: {missing[:10]}")
    weight = weight / denom[dst].clamp_min(1e-12)
    return SparseGraph(
        num_nodes=num_nodes,
        dst=dst,
        src=src,
        weight=weight,
        edge_type=torch.tensor([merged[k]["edge_type"] for k in keys], dtype=torch.uint8),
        reproj_error=torch.tensor([merged[k]["reproj_error"] for k in keys], dtype=torch.float32),
        cycle_error=torch.tensor([merged[k]["cycle_error"] for k in keys], dtype=torch.float32),
        visibility=torch.tensor([merged[k]["visibility"] for k in keys], dtype=torch.float32),
        confidence=torch.tensor([merged[k]["confidence"] for k in keys], dtype=torch.float32),
    )


def self_loop_graph(num_nodes: int, weight: float = 1.0) -> SparseGraph:
    """Return a normalized graph containing only self loops."""

    return merge_and_normalize(num_nodes, self_loop_candidates(num_nodes, weight))


def build_graph(
    num_nodes: int,
    candidates: list[EdgeCandidates],
    *,
    self_loop_weight: float = 1.0,
    frame_index: torch.Tensor | None = None,
    topk_cross_frame_edges: int | None = None,
) -> SparseGraph:
    all_candidates = concat_candidates([*candidates, self_loop_candidates(num_nodes, self_loop_weight)])
    if frame_index is not None and topk_cross_frame_edges is not None:
        all_candidates = topk_cross_frame(all_candidates, frame_index, topk_cross_frame_edges)
    return merge_and_normalize(num_nodes, all_candidates)


def mask_graph_edges(graph: SparseGraph, mask: torch.Tensor) -> SparseGraph:
    """Return a normalized graph containing the selected edges."""

    if mask.dtype != torch.bool:
        mask = mask.bool()
    candidates = EdgeCandidates(
        dst=graph.dst[mask],
        src=graph.src[mask],
        weight=graph.weight[mask],
        edge_type=graph.edge_type[mask],
        reproj_error=graph.reproj_error[mask],
        cycle_error=graph.cycle_error[mask],
        visibility=graph.visibility[mask],
        confidence=graph.confidence[mask],
    )
    return merge_and_normalize(graph.num_nodes, candidates)


def restrict_incoming_sources(
    graph: SparseGraph,
    *,
    dst_node: int,
    allowed_sources: torch.Tensor,
) -> SparseGraph:
    """Restrict one destination node to self-loop plus selected incoming sources."""

    allowed_sources = allowed_sources.to(device=graph.src.device, dtype=torch.long)
    edge_ids = torch.arange(graph.dst.numel(), device=graph.dst.device)
    is_dst = graph.dst == int(dst_node)
    is_allowed = torch.isin(graph.src, allowed_sources) | (graph.src == int(dst_node))
    keep = ~is_dst | is_allowed
    if not keep.any():
        raise ValueError("restricted graph would be empty")
    return mask_graph_edges(graph, keep[edge_ids])


def shuffled_endpoint_graph(graph: SparseGraph, *, seed: int = 0) -> SparseGraph:
    """Shuffle non-self source endpoints while preserving destinations and weights."""

    gen = torch.Generator(device="cpu").manual_seed(seed)
    is_self = graph.dst == graph.src
    cross_idx = torch.nonzero(~is_self.cpu(), as_tuple=False).flatten()
    src = graph.src.clone()
    if cross_idx.numel() > 1:
        perm = cross_idx[torch.randperm(cross_idx.numel(), generator=gen)]
        src[cross_idx] = graph.src[perm.to(device=graph.src.device)]
    candidates = EdgeCandidates(
        dst=graph.dst.clone(),
        src=src,
        weight=graph.weight.clone(),
        edge_type=graph.edge_type.clone(),
        reproj_error=graph.reproj_error.clone(),
        cycle_error=graph.cycle_error.clone(),
        visibility=graph.visibility.clone(),
        confidence=graph.confidence.clone(),
    )
    return merge_and_normalize(graph.num_nodes, candidates)


def random_same_degree_graph(graph: SparseGraph, *, seed: int = 0) -> SparseGraph:
    """Randomize non-self sources while preserving incoming degree and self loops."""

    gen = torch.Generator(device="cpu").manual_seed(seed)
    is_self = graph.dst == graph.src
    src = graph.src.clone()
    for edge_id in torch.nonzero(~is_self.cpu(), as_tuple=False).flatten().tolist():
        dst = int(graph.dst[edge_id])
        candidates = torch.arange(graph.num_nodes, dtype=torch.long)
        candidates = candidates[candidates != dst]
        src[edge_id] = candidates[torch.randint(candidates.numel(), (1,), generator=gen)]
    candidates = EdgeCandidates(
        dst=graph.dst.clone(),
        src=src,
        weight=graph.weight.clone(),
        edge_type=graph.edge_type.clone(),
        reproj_error=graph.reproj_error.clone(),
        cycle_error=graph.cycle_error.clone(),
        visibility=graph.visibility.clone(),
        confidence=graph.confidence.clone(),
    )
    return merge_and_normalize(graph.num_nodes, candidates)


def _mask_candidates(c: EdgeCandidates, mask: torch.Tensor) -> EdgeCandidates:
    return EdgeCandidates(
        c.dst[mask],
        c.src[mask],
        c.weight[mask],
        c.edge_type[mask],
        c.reproj_error[mask],
        c.cycle_error[mask],
        c.visibility[mask],
        c.confidence[mask],
    )


def _nanmin(a: float, b: float) -> float:
    if a != a:
        return b
    if b != b:
        return a
    return min(a, b)
