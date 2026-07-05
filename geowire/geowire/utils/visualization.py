from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch


def save_points_overlay(image_path: str | Path, points_xy: torch.Tensor, output_path: str | Path) -> None:
    image = plt.imread(str(image_path))
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image)
    pts = points_xy.detach().cpu()
    ax.scatter(pts[:, 0], pts[:, 1], s=8, c="lime")
    ax.set_axis_off()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
