from __future__ import annotations

import hashlib
import json
from pathlib import Path

from geowire.types import ClipRecord
from geowire.utils.io import iter_jsonl


def load_manifest(path: str | Path) -> list[ClipRecord]:
    records: list[ClipRecord] = []
    for row in iter_jsonl(path):
        records.append(
            ClipRecord(
                clip_id=row["clip_id"],
                scene_id=row["scene_id"],
                source_dataset=row["source_dataset"],
                frame_paths=tuple(row["frame_paths"]),
                frame_indices=tuple(row["frame_indices"]),
                timestamps_s=tuple(float(x) for x in row["timestamps_s"]),
                split=row["split"],
                question=row.get("question"),
                answer=row.get("answer"),
                task_type=row.get("task_type"),
                static_view_permutation_allowed=bool(row.get("static_view_permutation_allowed", False)),
                cache_dir=row.get("cache_dir"),
            )
        )
    return records


def manifest_hash(records: list[ClipRecord]) -> str:
    payload = [
        {
            "clip_id": r.clip_id,
            "scene_id": r.scene_id,
            "source_dataset": r.source_dataset,
            "frame_paths": r.frame_paths,
            "frame_indices": r.frame_indices,
            "timestamps_s": r.timestamps_s,
            "split": r.split,
        }
        for r in sorted(records, key=lambda x: x.clip_id)
    ]
    data = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()
