from __future__ import annotations

import argparse
from pathlib import Path

import torch

from _bootstrap import bootstrap

bootstrap()

from geowire.geometry.graph_builder import EdgeCandidates, build_graph
from geowire.training.pretrain_tip import run_tip_debug_step
from geowire.utils.io import write_json
from geowire.utils.reproducibility import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("runs/tip_debug"))
    args = parser.parse_args()
    seed_everything(3407)
    args.output.mkdir(parents=True, exist_ok=True)

    hidden = torch.randn(8, 16, requires_grad=True)
    clean = hidden.detach().clone()
    dst = torch.tensor([4, 5, 6, 7])
    src = torch.tensor([0, 1, 2, 3])
    ones = torch.ones(4)
    graph = build_graph(
        8,
        [
            EdgeCandidates(
                dst=dst,
                src=src,
                weight=ones,
                edge_type=torch.full((4,), 2, dtype=torch.uint8),
                reproj_error=torch.zeros(4),
                cycle_error=torch.zeros(4),
                visibility=ones,
                confidence=ones,
            )
        ],
    )
    metrics = run_tip_debug_step(hidden, clean, graph)
    write_json(args.output / "metrics.json", metrics)
    print(metrics)


if __name__ == "__main__":
    main()
