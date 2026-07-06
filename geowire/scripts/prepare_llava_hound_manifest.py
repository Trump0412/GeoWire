from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from _bootstrap import bootstrap

bootstrap()

from geowire.utils.io import write_json, write_jsonl

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def natural_key(path: Path) -> tuple[Any, ...]:
    parts = re.split(r"(\d+)", path.name)
    return tuple(int(part) if part.isdigit() else part.lower() for part in parts)


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


def sample_indices(count: int, max_frames: int) -> list[int]:
    if count <= 0:
        return []
    if count <= max_frames:
        return list(range(count))
    if max_frames <= 1:
        return [0]
    return sorted({round(i * (count - 1) / (max_frames - 1)) for i in range(max_frames)})


def frame_files(frame_dir: Path) -> list[Path]:
    if not frame_dir.is_dir():
        return []
    return sorted(
        [path for path in frame_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS],
        key=natural_key,
    )


def build_manifest_rows(
    items: list[dict[str, Any]],
    *,
    media_root: Path,
    max_frames: int,
    trust_frame_count: int,
    limit: int | None,
    offset: int,
    split: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    stats = {
        "input_items": len(items),
        "offset": offset,
        "limit": limit,
        "trust_frame_count": trust_frame_count,
        "kept": 0,
        "skipped_missing_frames": 0,
        "skipped_missing_qa": 0,
    }
    for item in items[offset:]:
        if limit is not None and len(rows) >= limit:
            break
        question, answer = first_qa(item.get("conversations", []))
        if not question or not answer:
            stats["skipped_missing_qa"] += 1
            continue
        video_rel = item.get("video")
        if not video_rel:
            stats["skipped_missing_frames"] += 1
            continue
        video_dir = media_root / video_rel
        if trust_frame_count > 0:
            frames = [video_dir / f"frame_{index + 1:02d}.jpg" for index in range(trust_frame_count)]
        else:
            frames = frame_files(video_dir)
        if len(frames) < 2:
            stats["skipped_missing_frames"] += 1
            continue
        indices = sample_indices(len(frames), max_frames)
        selected = [frames[index] for index in indices]
        clip_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(item.get("id", video_rel))).strip("_")
        scene_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(video_rel).rstrip("/").split("/")[-1]).strip("_")
        rows.append(
            {
                "clip_id": clip_id,
                "scene_id": scene_id,
                "source_dataset": "llava_hound",
                "frame_paths": [str(path) for path in selected],
                "frame_indices": indices,
                "timestamps_s": [float(index) for index in indices],
                "split": split,
                "question": question,
                "answer": answer,
                "task_type": item.get("tag") or "video_qa",
                "static_view_permutation_allowed": False,
            }
        )
    stats["kept"] = len(rows)
    return rows, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument(
        "--trust-frame-count",
        type=int,
        default=0,
        help="Fast path for already-extracted frame dirs named frame_01.jpg...frame_NN.jpg; skips directory listing.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--split", choices=["train", "val", "test"], default="train")
    parser.add_argument("--write-report", type=Path)
    args = parser.parse_args()

    with args.input.open("r", encoding="utf-8") as handle:
        items = json.load(handle)
    if not isinstance(items, list):
        raise TypeError(f"expected list JSON at {args.input}")
    rows, stats = build_manifest_rows(
        items,
        media_root=args.media_root,
        max_frames=args.max_frames,
        trust_frame_count=args.trust_frame_count,
        limit=args.limit,
        offset=args.offset,
        split=args.split,
    )
    write_jsonl(args.output, rows)
    report = {"output": str(args.output), "media_root": str(args.media_root), **stats}
    if args.write_report:
        write_json(args.write_report, report)
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
