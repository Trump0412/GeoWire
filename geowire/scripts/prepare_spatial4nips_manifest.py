from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

from _bootstrap import bootstrap

bootstrap()

from geowire.utils.io import write_json, write_jsonl

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def clean_prompt(text: str) -> str:
    text = text.replace("<video>", "").replace("<image>", "")
    return "\n".join(line.strip() for line in text.splitlines() if line.strip()).strip()


def first_qa(conversations: list[dict[str, str]]) -> tuple[str | None, str | None]:
    question: str | None = None
    for turn in conversations:
        role = turn.get("from")
        value = turn.get("value", "")
        if role == "human" and question is None:
            question = clean_prompt(value)
        elif role == "gpt" and question is not None:
            return question, value.strip()
    return None, None


def sanitize(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return value[:180] or "unknown"


def natural_key(path: Path) -> tuple[Any, ...]:
    parts = re.split(r"(\d+)", path.name)
    return tuple(int(part) if part.isdigit() else part.lower() for part in parts)


def frame_files(frame_dir: Path) -> list[Path]:
    if not frame_dir.is_dir():
        return []
    return sorted(
        [path for path in frame_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS],
        key=natural_key,
    )


def sample_indices(count: int, max_frames: int) -> list[int]:
    if count <= 0:
        return []
    if count <= max_frames:
        return list(range(count))
    if max_frames <= 1:
        return [0]
    return sorted({round(i * (count - 1) / (max_frames - 1)) for i in range(max_frames)})


def iter_input_records(path: Path) -> Iterable[dict[str, Any]]:
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
        return
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise TypeError(f"expected list JSON at {path}")
    yield from data


def selected_media_paths(
    item: dict[str, Any],
    *,
    media_root: Path,
    max_frames: int,
    trust_video_frame_count: int,
) -> tuple[list[Path], list[int]]:
    if item.get("images"):
        images = [media_root / path for path in item["images"]]
        indices = sample_indices(len(images), max_frames)
        return [images[index] for index in indices], indices

    video_rel = item.get("video")
    if not video_rel:
        return [], []
    video_dir = media_root / video_rel
    if trust_video_frame_count > 0:
        frames = [video_dir / f"frame_{index + 1:02d}.jpg" for index in range(trust_video_frame_count)]
    else:
        frames = frame_files(video_dir)
    indices = sample_indices(len(frames), max_frames)
    return [frames[index] for index in indices], indices


def scene_id_for_item(item: dict[str, Any], frame_paths: list[Path]) -> str:
    if item.get("scene_name"):
        return sanitize(str(item["scene_name"]))
    if item.get("video"):
        return sanitize(str(item["video"]).rstrip("/").split("/")[-1])
    if frame_paths:
        parts = frame_paths[0].parts
        for marker in ("scannet", "scannetpp", "arkitscenes", "structured3d", "MindCube"):
            if marker in parts:
                idx = parts.index(marker)
                if idx + 1 < len(parts):
                    return sanitize(parts[idx + 1])
        return sanitize(frame_paths[0].parent.name)
    return sanitize(str(item.get("id", "unknown")))


def clip_id_for_media(source_dataset: str, frame_paths: list[Path]) -> str:
    media_key = "\n".join(str(path) for path in frame_paths)
    digest = hashlib.sha1(media_key.encode("utf-8")).hexdigest()[:20]
    return f"{sanitize(source_dataset)}_{digest}"


def build_manifest_rows(
    input_paths: list[Path],
    *,
    source_dataset: str,
    media_root: Path,
    max_frames: int,
    min_frames: int,
    trust_video_frame_count: int,
    split: str,
    limit: int | None,
    check_files: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    stats: dict[str, Any] = {
        "source_dataset": source_dataset,
        "input_paths": [str(path) for path in input_paths],
        "max_frames": max_frames,
        "min_frames": min_frames,
        "records_seen": 0,
        "kept": 0,
        "unique_clip_ids": 0,
        "skipped_missing_qa": 0,
        "skipped_too_few_frames": 0,
        "skipped_missing_files": 0,
    }
    unique_clip_ids: set[str] = set()
    for input_path in input_paths:
        for item in iter_input_records(input_path):
            stats["records_seen"] += 1
            if limit is not None and len(rows) >= limit:
                break
            question, answer = first_qa(item.get("conversations", []))
            if not question or not answer:
                stats["skipped_missing_qa"] += 1
                continue
            frame_paths, frame_indices = selected_media_paths(
                item,
                media_root=media_root,
                max_frames=max_frames,
                trust_video_frame_count=trust_video_frame_count,
            )
            if len(frame_paths) < min_frames:
                stats["skipped_too_few_frames"] += 1
                continue
            if check_files and not all(path.exists() for path in frame_paths):
                stats["skipped_missing_files"] += 1
                continue
            clip_id = clip_id_for_media(source_dataset, frame_paths)
            unique_clip_ids.add(clip_id)
            rows.append(
                {
                    "clip_id": clip_id,
                    "scene_id": scene_id_for_item(item, frame_paths),
                    "source_dataset": source_dataset,
                    "frame_paths": [str(path) for path in frame_paths],
                    "frame_indices": frame_indices,
                    "timestamps_s": [float(index) for index in frame_indices],
                    "split": split,
                    "question": question,
                    "answer": answer,
                    "task_type": item.get("question_type") or item.get("tag") or item.get("data_source") or "spatial_qa",
                    "static_view_permutation_allowed": len(frame_paths) > 1 and "vst" not in source_dataset,
                    "raw_id": item.get("id"),
                    "data_source": item.get("data_source"),
                }
            )
        if limit is not None and len(rows) >= limit:
            break
    stats["kept"] = len(rows)
    stats["unique_clip_ids"] = len(unique_clip_ids)
    return rows, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--source-dataset", required=True)
    parser.add_argument("--media-root", type=Path, default=Path("data/media"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument("--min-frames", type=int, default=1)
    parser.add_argument("--trust-video-frame-count", type=int, default=0)
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--check-files", action="store_true")
    parser.add_argument("--write-report", type=Path)
    args = parser.parse_args()

    rows, stats = build_manifest_rows(
        args.input,
        source_dataset=args.source_dataset,
        media_root=args.media_root,
        max_frames=args.max_frames,
        min_frames=args.min_frames,
        trust_video_frame_count=args.trust_video_frame_count,
        split=args.split,
        limit=args.limit,
        check_files=args.check_files,
    )
    write_jsonl(args.output, rows)
    report = {"output": str(args.output), **stats}
    if args.write_report:
        write_json(args.write_report, report)
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
