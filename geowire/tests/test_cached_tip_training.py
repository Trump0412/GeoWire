from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_cached_tip_training_smoke(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    assets = tmp_path / "toy_scene"
    manifest = assets / "manifest.jsonl"
    cache = tmp_path / "cache"
    graph_out = tmp_path / "graphs"
    out = tmp_path / "tip"

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
            "scripts/train_tip.py",
            "--manifest",
            str(manifest),
            "--cache-root",
            str(cache),
            "--output",
            str(out),
            "--steps",
            "1",
        ],
        cwd=root,
        check=True,
    )

    assert (out / "geowire_adapter.pt").exists()
    assert (out / "metrics.json").exists()
    assert (out / "metrics.jsonl").exists()
