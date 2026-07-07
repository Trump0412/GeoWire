from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import torch

from _bootstrap import bootstrap

bootstrap()

from geowire.constants import EDGE_PROJECTIVE
from geowire.geometry.graph_builder import EdgeCandidates, build_graph
from geowire.geometry.graph_io import save_graph_npz
from geowire.geometry.qwen_layout import QwenTokenLayoutBuilder
from geowire.geometry.vggt_cache import save_frame_transforms, save_token_layout, write_cache_metadata
from geowire.models.qwen3vl_cache import _frame_transforms_from_grid, ordered_image_messages
from geowire.types import ClipRecord
from geowire.utils.io import iter_jsonl
from geowire.utils.io import write_json


def clip_shard(clip_id: str, num_shards: int) -> int:
    if num_shards <= 1:
        return 0
    digest = hashlib.sha1(clip_id.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % num_shards


def iter_manifest(path: Path):
    for row in iter_jsonl(path):
        yield ClipRecord(
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


def patch_and_merge(config, processor) -> tuple[int, int, int]:
    vision_config = getattr(config, "vision_config", None)
    text_config = getattr(config, "text_config", None)
    patch = getattr(vision_config, "patch_size", None) or getattr(getattr(processor, "image_processor", None), "patch_size", 16)
    merge = getattr(vision_config, "spatial_merge_size", None) or getattr(getattr(processor, "image_processor", None), "merge_size", 2)
    hidden = getattr(text_config, "hidden_size", None) or getattr(config, "hidden_size", None)
    if hidden is None:
        raise RuntimeError("cannot infer Qwen hidden size from config")
    return int(patch), int(merge), int(hidden)


def collect_image_inputs(messages):
    from qwen_vl_utils import process_vision_info

    image_inputs, video_inputs = process_vision_info(messages)
    if video_inputs:
        raise ValueError("GeoWire cache uses ordered images, not Qwen video mode")
    return image_inputs


def grid_correspondence_candidates(layout, *, all_pairs: bool = False) -> EdgeCandidates:
    dst: list[int] = []
    src: list[int] = []
    weight: list[float] = []
    offsets = [int(x) for x in layout.token_offsets.tolist()]
    frame_lengths = [offsets[i + 1] - offsets[i] for i in range(layout.num_frames)]
    for dst_frame in range(layout.num_frames):
        if all_pairs:
            source_frames = [src_frame for src_frame in range(layout.num_frames) if src_frame != dst_frame]
        else:
            source_frames = [src_frame for src_frame in (dst_frame - 1, dst_frame + 1) if 0 <= src_frame < layout.num_frames]
        for src_frame in source_frames:
            count = min(frame_lengths[dst_frame], frame_lengths[src_frame])
            if count <= 0:
                continue
            decay = 1.0 / float(1 + abs(dst_frame - src_frame))
            for local in range(count):
                dst_id = offsets[dst_frame] + local
                src_id = offsets[src_frame] + local
                if bool(layout.valid[dst_id]) and bool(layout.valid[src_id]):
                    dst.append(dst_id)
                    src.append(src_id)
                    weight.append(decay)
    if not dst:
        empty_long = torch.empty(0, dtype=torch.long)
        empty_float = torch.empty(0, dtype=torch.float32)
        return EdgeCandidates(
            dst=empty_long,
            src=empty_long,
            weight=empty_float,
            edge_type=torch.empty(0, dtype=torch.uint8),
            reproj_error=empty_float,
            cycle_error=empty_float,
            visibility=empty_float,
            confidence=empty_float,
        )
    n = len(dst)
    return EdgeCandidates(
        dst=torch.tensor(dst, dtype=torch.long),
        src=torch.tensor(src, dtype=torch.long),
        weight=torch.tensor(weight, dtype=torch.float32),
        edge_type=torch.full((n,), EDGE_PROJECTIVE, dtype=torch.uint8),
        reproj_error=torch.zeros(n, dtype=torch.float32),
        cycle_error=torch.zeros(n, dtype=torch.float32),
        visibility=torch.ones(n, dtype=torch.float32),
        confidence=torch.ones(n, dtype=torch.float32),
    )


def cache_record(
    record,
    *,
    processor,
    config,
    cache_root: Path,
    qwen_checkpoint: str,
    all_pairs: bool,
    topk_cross_frame: int,
) -> None:
    patch, merge, hidden_size = patch_and_merge(config, processor)
    messages = ordered_image_messages(record)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs = collect_image_inputs(messages)
    inputs = processor(text=[text], images=image_inputs, videos=None, padding=True, return_tensors="pt")
    grid_thw = inputs["image_grid_thw"].detach().cpu().long()
    frame_transforms = _frame_transforms_from_grid(record, grid_thw, patch, True)
    grid_hw = tuple((int(thw[1].item() // merge), int(thw[2].item() // merge)) for thw in grid_thw)
    layout = QwenTokenLayoutBuilder(hidden_size=hidden_size).build(frame_transforms, grid_hw)
    graph = build_graph(
        int(layout.frame_index.numel()),
        [grid_correspondence_candidates(layout, all_pairs=all_pairs)],
        frame_index=layout.frame_index,
        topk_cross_frame_edges=topk_cross_frame,
    )

    clip_dir = cache_root / record.clip_id
    clip_dir.mkdir(parents=True, exist_ok=True)
    write_cache_metadata(
        clip_dir,
        {
            "clip_id": record.clip_id,
            "source_dataset": record.source_dataset,
            "backend": "qwen_layout_grid",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "qwen_checkpoint": qwen_checkpoint,
            "graph_backend": "same_grid_correspondence",
            "all_pairs": all_pairs,
            "topk_cross_frame": topk_cross_frame,
            "image_max_pixels": os.environ.get("GEOWIRE_IMAGE_MAX_PIXELS"),
            "image_min_pixels": os.environ.get("GEOWIRE_IMAGE_MIN_PIXELS"),
            "note": "Compact training cache: exact Qwen token layout plus deterministic same-grid cross-frame graph.",
        },
    )
    write_json(clip_dir / "source_record.json", record.__dict__)
    write_json(
        clip_dir / "qwen_input.json",
        {
            "input_mode": "ordered_images",
            "image_grid_thw": grid_thw.tolist(),
            "patch_size": patch,
            "spatial_merge_size": merge,
        },
    )
    save_frame_transforms(clip_dir / "frame_transforms.json", frame_transforms)
    save_token_layout(clip_dir / "token_layout.safetensors", layout)
    save_graph_npz(clip_dir / "graph_coo.npz", graph)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--qwen-checkpoint", default="/mnt/guojh/lq/new/models/Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--all-pairs", action="store_true")
    parser.add_argument("--topk-cross-frame", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args()

    if not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("--shard-index must be in [0, --num-shards)")

    from transformers import AutoConfig, AutoProcessor

    processor = AutoProcessor.from_pretrained(args.qwen_checkpoint, trust_remote_code=True)
    config = AutoConfig.from_pretrained(args.qwen_checkpoint, trust_remote_code=True)
    seen: set[str] = set()
    processed = 0
    skipped_existing = 0
    skipped_shard = 0
    for record in iter_manifest(args.manifest):
        if record.clip_id in seen:
            continue
        seen.add(record.clip_id)
        if clip_shard(record.clip_id, args.num_shards) != args.shard_index:
            skipped_shard += 1
            continue
        clip_dir = args.cache_root / record.clip_id
        if args.skip_existing and (clip_dir / "token_layout.safetensors").exists() and (clip_dir / "graph_coo.npz").exists():
            skipped_existing += 1
            continue
        cache_record(
            record,
            processor=processor,
            config=config,
            cache_root=args.cache_root,
            qwen_checkpoint=args.qwen_checkpoint,
            all_pairs=args.all_pairs,
            topk_cross_frame=args.topk_cross_frame,
        )
        processed += 1
        if processed % args.log_every == 0:
            print(
                json.dumps(
                    {
                        "processed": processed,
                        "seen_unique": len(seen),
                        "skipped_shard": skipped_shard,
                        "skipped_existing": skipped_existing,
                        "shard": args.shard_index,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if args.limit is not None and processed >= args.limit:
            break

    args.cache_root.mkdir(parents=True, exist_ok=True)
    write_json(
        args.cache_root / f"cache_index_shard{args.shard_index:03d}.json",
        {
            "manifest": str(args.manifest),
            "cache_root": str(args.cache_root),
            "backend": "qwen_layout_grid",
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
            "processed": processed,
            "seen_unique": len(seen),
            "skipped_shard": skipped_shard,
            "skipped_existing": skipped_existing,
        },
    )
    print(json.dumps({"processed": processed, "shard": args.shard_index}, sort_keys=True))


if __name__ == "__main__":
    main()
