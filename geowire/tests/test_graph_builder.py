from __future__ import annotations

import torch

from geowire.constants import EDGE_TRACK
from geowire.geometry.graph_builder import (
    EdgeCandidates,
    build_graph,
    random_same_degree_graph,
    restrict_incoming_sources,
    self_loop_graph,
    shuffled_endpoint_graph,
)


def make_candidates() -> EdgeCandidates:
    dst = torch.tensor([1, 1, 2])
    src = torch.tensor([0, 0, 1])
    return EdgeCandidates(
        dst=dst,
        src=src,
        weight=torch.tensor([0.5, 0.5, 2.0]),
        edge_type=torch.full((3,), EDGE_TRACK, dtype=torch.uint8),
        reproj_error=torch.zeros(3),
        cycle_error=torch.zeros(3),
        visibility=torch.ones(3),
        confidence=torch.ones(3),
    )


def test_graph_self_loops_and_normalization() -> None:
    graph = build_graph(3, [make_candidates()])
    row_sum = torch.zeros(3)
    row_sum.index_add_(0, graph.dst, graph.weight)
    assert torch.allclose(row_sum, torch.ones(3))
    assert graph.dst.numel() == 5


def test_topk_cross_frame() -> None:
    cand = EdgeCandidates(
        dst=torch.tensor([3, 3, 3]),
        src=torch.tensor([0, 1, 2]),
        weight=torch.tensor([0.1, 0.9, 0.2]),
        edge_type=torch.full((3,), EDGE_TRACK, dtype=torch.uint8),
        reproj_error=torch.zeros(3),
        cycle_error=torch.zeros(3),
        visibility=torch.ones(3),
        confidence=torch.ones(3),
    )
    frame_index = torch.tensor([0, 0, 0, 1])
    graph = build_graph(4, [cand], frame_index=frame_index, topk_cross_frame_edges=1)
    cross = graph.dst != graph.src
    assert graph.src[cross].tolist() == [1]


def test_self_loop_graph() -> None:
    graph = self_loop_graph(4)
    assert graph.dst.tolist() == [0, 1, 2, 3]
    assert graph.src.tolist() == [0, 1, 2, 3]
    assert torch.allclose(graph.weight, torch.ones(4))


def test_restrict_incoming_sources_keeps_other_nodes_normalized() -> None:
    graph = build_graph(3, [make_candidates()])
    restricted = restrict_incoming_sources(graph, dst_node=1, allowed_sources=torch.tensor([0]))
    row_sum = torch.zeros(3)
    row_sum.index_add_(0, restricted.dst, restricted.weight)
    assert torch.allclose(row_sum, torch.ones(3))
    assert set(restricted.src[restricted.dst == 1].tolist()) <= {0, 1}


def test_random_and_shuffled_graphs_remain_normalized() -> None:
    graph = build_graph(3, [make_candidates()])
    for variant in (random_same_degree_graph(graph), shuffled_endpoint_graph(graph)):
        row_sum = torch.zeros(3)
        row_sum.index_add_(0, variant.dst, variant.weight)
        assert torch.allclose(row_sum, torch.ones(3))
