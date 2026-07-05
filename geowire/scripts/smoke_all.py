from __future__ import annotations

import argparse
import json
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
            "scripts/check_training_readiness.py",
            "--phase",
            "phase1",
            "--manifest",
            "assets/toy_scene/manifest.jsonl",
            "--cache-root",
            str(args.output / "toy_cache"),
            "--write",
            str(args.output / "phase1_readiness.json"),
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
    run(
        [
            py,
            "scripts/train_tip.py",
            "--manifest",
            "assets/toy_scene/manifest.jsonl",
            "--cache-root",
            str(args.output / "toy_cache"),
            "--output",
            str(args.output / "tip_debug"),
            "--steps",
            "2",
        ]
    )
    pred_path = args.output / "toy_predictions.jsonl"
    pred_path.write_text(
        "\n".join(
            [
                json.dumps({"id": "toy_0", "prediction": "A", "answer": "A"}),
                json.dumps({"id": "toy_1", "prediction": "B", "answer": "C"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    run(
        [
            py,
            "scripts/evaluate.py",
            "--config",
            "configs/eval_vsi.yaml",
            "--predictions",
            str(pred_path),
            "--write",
            str(args.output / "eval_report.json"),
        ]
    )
    if not args.skip_tests:
        run([py, "-m", "pytest", "-q"])
    print(args.output)


if __name__ == "__main__":
    main()
