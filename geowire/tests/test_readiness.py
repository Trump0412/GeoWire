from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_phase1_readiness_passes_for_toy_cache(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    assets = tmp_path / "toy_scene"
    manifest = assets / "manifest.jsonl"
    cache = tmp_path / "cache"
    graph_out = tmp_path / "graphs"
    report = tmp_path / "readiness.json"

    subprocess.run(
        [sys.executable, "scripts/generate_toy_scene.py", "--output", str(assets), "--manifest", str(manifest)],
        cwd=root,
        check=True,
    )
    subprocess.run(
        [sys.executable, "scripts/cache_vggt.py", "--manifest", str(manifest), "--cache-root", str(cache)],
        cwd=root,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/build_graphs.py",
            "--manifest",
            str(manifest),
            "--cache-root",
            str(cache),
            "--output",
            str(graph_out),
        ],
        cwd=root,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/check_training_readiness.py",
            "--phase",
            "phase1",
            "--manifest",
            str(manifest),
            "--cache-root",
            str(cache),
            "--write",
            str(report),
        ],
        cwd=root,
        check=True,
    )
    assert report.exists()
