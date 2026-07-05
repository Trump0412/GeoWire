from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from geowire.types import SparseGraph


def save_graph_npz(path: str | Path, graph: SparseGraph) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        dst=graph.dst.cpu().numpy(),
        src=graph.src.cpu().numpy(),
        weight=graph.weight.cpu().numpy(),
        edge_type=graph.edge_type.cpu().numpy(),
        reproj_error=graph.reproj_error.cpu().numpy(),
        cycle_error=graph.cycle_error.cpu().numpy(),
        visibility=graph.visibility.cpu().numpy(),
        confidence=graph.confidence.cpu().numpy(),
        num_nodes=np.array([graph.num_nodes], dtype=np.int64),
    )


def load_graph_npz(path: str | Path) -> SparseGraph:
    data = np.load(path)
    return SparseGraph(
        num_nodes=int(data["num_nodes"][0]),
        dst=torch.from_numpy(data["dst"]).long(),
        src=torch.from_numpy(data["src"]).long(),
        weight=torch.from_numpy(data["weight"]).float(),
        edge_type=torch.from_numpy(data["edge_type"]).to(torch.uint8),
        reproj_error=torch.from_numpy(data["reproj_error"]).float(),
        cycle_error=torch.from_numpy(data["cycle_error"]).float(),
        visibility=torch.from_numpy(data["visibility"]).float(),
        confidence=torch.from_numpy(data["confidence"]).float(),
    )
