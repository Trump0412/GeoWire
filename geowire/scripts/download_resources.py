from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download

from _bootstrap import bootstrap

bootstrap()

from geowire.constants import HF_MIRROR_ENDPOINT


def snapshot(repo_id: str, local_dir: Path, repo_type: str, endpoint: str) -> str:
    local_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_ENDPOINT", endpoint)
    return snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
        endpoint=endpoint,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="kind", required=True)

    model = sub.add_parser("model")
    model.add_argument("repo_id")
    model.add_argument("--local-dir", type=Path, required=True)
    model.add_argument("--endpoint", default=HF_MIRROR_ENDPOINT)

    dataset = sub.add_parser("dataset")
    dataset.add_argument("repo_id")
    dataset.add_argument("--local-dir", type=Path, required=True)
    dataset.add_argument("--endpoint", default=HF_MIRROR_ENDPOINT)

    args = parser.parse_args()
    repo_type = "model" if args.kind == "model" else "dataset"
    path = snapshot(args.repo_id, args.local_dir, repo_type, args.endpoint)
    print(path)


if __name__ == "__main__":
    main()
