from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from _bootstrap import bootstrap


ROOT = bootstrap()


def run(cmd: list[str]) -> None:
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(ROOT))
    env.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("runs/smoke"))
    parser.add_argument("--skip-tests", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    py = sys.executable
    run([py, "scripts/inspect_environment.py", "--write", str(args.output / "env_report.json")])
    run(
        [
            py,
            "scripts/generate_toy_scene.py",
            "--output",
            "assets/toy_scene",
            "--manifest",
            "assets/toy_scene/manifest.jsonl",
        ]
    )
    run([py, "scripts/verify_coordinate_contract.py", "--write", str(args.output / "coordinate_contract.json")])
    run(
        [
            py,
            "scripts/cache_vggt.py",
            "--manifest",
            "assets/toy_scene/manifest.jsonl",
            "--cache-root",
            str(args.output / "toy_cache"),
            "--backend",
            "toy",
        ]
    )
    run(
        [
            py,
            "scripts/build_graphs.py",
            "--manifest",
            "assets/toy_scene/manifest.jsonl",
            "--cache-root",
            str(args.output / "toy_cache"),
            "--output",
            str(args.output / "phase0_graph"),
        ]
    )
    run(
        [
            py,
            "scripts/verify_graph.py",
            "--graph",
            str(args.output / "toy_cache" / "toy_scene_clip_000" / "graph_coo.npz"),
        ]
    )
    run([py, "scripts/train_tip.py", "--output", str(args.output / "tip_debug")])
    if not args.skip_tests:
        run([py, "-m", "pytest", "-q"])
    print(args.output)


if __name__ == "__main__":
    main()
