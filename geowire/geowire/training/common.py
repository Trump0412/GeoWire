from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path

from geowire.utils.io import write_json


def write_run_metadata(output_dir: str | Path, metadata: dict) -> None:
    serializable = {}
    for key, value in metadata.items():
        serializable[key] = asdict(value) if is_dataclass(value) else value
    write_json(Path(output_dir) / "metadata.json", serializable)
