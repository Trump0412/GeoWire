from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch


def save_edge_scatter(points_xy: torch.Tensor, output_path: str | Path) -> None:
    pts = points_xy.detach().cpu()
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(pts[:, 0], pts[:, 1], s=8)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
