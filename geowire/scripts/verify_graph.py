from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import bootstrap

bootstrap()

from geowire.geometry.graph_io import load_graph_npz
from geowire.geometry.graph_stats import graph_stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=Path, required=True)
    args = parser.parse_args()
    graph = load_graph_npz(args.graph)
    stats = graph_stats(graph)
    print(json.dumps(stats, indent=2))
    if abs(stats["row_sum_min"] - 1.0) > 1e-5 or abs(stats["row_sum_max"] - 1.0) > 1e-5:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
