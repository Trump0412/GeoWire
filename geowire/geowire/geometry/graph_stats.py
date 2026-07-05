from __future__ import annotations

import torch

from geowire.constants import EDGE_PROJECTIVE, EDGE_SELF, EDGE_TRACK
from geowire.types import SparseGraph


def graph_stats(graph: SparseGraph) -> dict[str, float | int]:
    row_sum = torch.zeros(graph.num_nodes, dtype=torch.float32)
    row_sum.index_add_(0, graph.dst.cpu(), graph.weight.cpu())
    return {
        "num_nodes": graph.num_nodes,
        "num_edges": int(graph.dst.numel()),
        "self_edges": int(((graph.edge_type & EDGE_SELF) != 0).sum()),
        "track_edges": int(((graph.edge_type & EDGE_TRACK) != 0).sum()),
        "projective_edges": int(((graph.edge_type & EDGE_PROJECTIVE) != 0).sum()),
        "row_sum_min": float(row_sum.min()),
        "row_sum_max": float(row_sum.max()),
    }
