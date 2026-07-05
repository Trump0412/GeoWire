from __future__ import annotations

import torch

from geowire.data.tip_dataset import sample_tip_targets
from geowire.geometry.graph_builder import EdgeCandidates, build_graph


def test_sample_tip_targets_requires_cross_frame_neighbors() -> None:
    ones = torch.ones(3)
    graph = build_graph(
        4,
        [
            EdgeCandidates(
                dst=torch.tensor([2, 2, 3]),
                src=torch.tensor([0, 1, 1]),
                weight=ones,
                edge_type=torch.full((3,), 2, dtype=torch.uint8),
                reproj_error=torch.zeros(3),
                cycle_error=torch.zeros(3),
                visibility=ones,
                confidence=ones,
            )
        ],
    )
    frame_index = torch.tensor([0, 1, 2, 2])
    targets = sample_tip_targets(graph, frame_index, mask_ratio=1.0)
    assert 2 in targets.masked_node_ids.tolist()
    assert len(targets.equivalent_support_groups) == 1
