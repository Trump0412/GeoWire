from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import torch
from PIL import Image

from _bootstrap import bootstrap

bootstrap()

from geowire.data.manifest import load_manifest
from geowire.geometry.qwen_layout import QwenTokenLayoutBuilder
from geowire.geometry.transforms import make_frame_transform
from geowire.geometry.vggt_cache import (
    save_frame_transforms,
    save_semantic_tokens,
    save_token_layout,
    save_vggt_geometry,
    write_cache_metadata,
)
from geowire.geometry.vggt_provider import VGGTProvider, attach_tracks
from geowire.models.qwen3vl_cache import extract_qwen_visual_cache, load_qwen_processor_and_model
from geowire.utils.io import write_json


def grid_for_size(size_wh: tuple[int, int], patch: int = 32) -> tuple[int, int]:
    w, h = size_wh
    return max(1, h // patch), max(1, w // patch)


def synthetic_semantic_tokens(layout, hidden_size: int) -> torch.Tensor:
    """Build deterministic toy semantic tokens from geometry fields."""

    xy = layout.center_raw_xy.float()
    xy = xy / xy.max(dim=0).values.clamp_min(1.0)
    frame = layout.frame_index.float().unsqueeze(-1) / max(layout.num_frames - 1, 1)
    row = layout.grid_row.float().unsqueeze(-1) / layout.grid_row.max().clamp_min(1).float()
    col = layout.grid_col.float().unsqueeze(-1) / layout.grid_col.max().clamp_min(1).float()
    base = torch.cat(
        [
            xy,
            torch.sin(xy * torch.pi),
            torch.cos(xy * torch.pi),
            frame,
            row,
            col,
            layout.valid.float().unsqueeze(-1),
        ],
        dim=-1,
    )
    repeats = (hidden_size + base.shape[-1] - 1) // base.shape[-1]
    hidden = base.repeat(1, repeats)[:, :hidden_size]
    hidden = hidden - hidden.mean(dim=-1, keepdim=True)
    hidden = hidden / hidden.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    hidden = torch.where(layout.valid.unsqueeze(-1), hidden, torch.zeros_like(hidden))
    return hidden


def cache_toy_manifest(manifest: Path, cache_root: Path, *, hidden_size: int) -> None:
    records = load_manifest(manifest)
    for record in records:
        clip_dir = cache_root / record.clip_id
        clip_dir.mkdir(parents=True, exist_ok=True)
        raw_sizes: list[tuple[int, int]] = []
        for frame_path in record.frame_paths:
            with Image.open(frame_path) as img:
                raw_sizes.append(img.size)
        frame_transforms = tuple(
            make_frame_transform(i, raw_size, (448, 448), (518, 518))
            for i, raw_size in enumerate(raw_sizes)
        )
        grid_hw = tuple(grid_for_size(ft.qwen_size_wh) for ft in frame_transforms)
        layout = QwenTokenLayoutBuilder(hidden_size=hidden_size).build(frame_transforms, grid_hw)

        write_cache_metadata(
            clip_dir,
            {
                "clip_id": record.clip_id,
                "source_dataset": record.source_dataset,
                "backend": "toy",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "note": "Synthetic cache: valid for software smoke only, not VGGT geometry claims.",
            },
        )
        write_json(clip_dir / "source_record.json", record.__dict__)
        save_frame_transforms(clip_dir / "frame_transforms.json", frame_transforms)
        write_json(
            clip_dir / "qwen_input.json",
            {"input_mode": "ordered_images", "grid_hw": [list(x) for x in grid_hw]},
        )
        save_token_layout(clip_dir / "token_layout.safetensors", layout)
        save_semantic_tokens(clip_dir / "semantic_tokens.safetensors", synthetic_semantic_tokens(layout, hidden_size))
    write_json(
        cache_root / "cache_index.json",
        {
            "backend": "toy",
            "manifest": str(manifest),
            "num_clips": len(records),
            "clip_ids": [r.clip_id for r in records],
        },
    )
    print(cache_root)


def select_anchor_frames(num_frames: int, strategy: str, count: int) -> list[int]:
    if num_frames <= 0:
        return []
    if strategy == "first":
        return [0]
    if strategy == "all":
        return list(range(num_frames))
    if count <= 1:
        return [0]
    idx = torch.linspace(0, num_frames - 1, steps=min(count, num_frames)).round().long().tolist()
    return sorted(set(int(x) for x in idx))


def select_query_tokens(layout, anchor_frame: int, max_tokens: int) -> torch.Tensor:
    ids = torch.nonzero((layout.frame_index == int(anchor_frame)) & layout.valid, as_tuple=False).flatten().long()
    if max_tokens > 0 and ids.numel() > max_tokens:
        pick = torch.linspace(0, ids.numel() - 1, steps=max_tokens).round().long()
        ids = ids[pick]
    return ids


def cache_real_manifest(
    manifest: Path,
    cache_root: Path,
    *,
    qwen_checkpoint: str,
    vggt_checkpoint: str,
    vggt_source: Path,
    device: str,
    dtype: torch.dtype,
    track_anchor_strategy: str,
    num_track_anchors: int,
    max_track_query_tokens: int,
) -> None:
    records = load_manifest(manifest)
    processor, qwen_model = load_qwen_processor_and_model(qwen_checkpoint, device=device, dtype=dtype)
    vggt = VGGTProvider(vggt_checkpoint, source_path=vggt_source, device=device, dtype=dtype)

    for record in records:
        clip_dir = cache_root / record.clip_id
        clip_dir.mkdir(parents=True, exist_ok=True)
        qwen_cache = extract_qwen_visual_cache(record, processor=processor, model=qwen_model)
        geometry = vggt.infer_geometry(list(record.frame_paths), frame_transforms=qwen_cache.frame_transforms)

        track_xy_rows = []
        track_vis_rows = []
        track_conf_rows = []
        anchor_rows = []
        query_id_rows = []
        for anchor in select_anchor_frames(len(record.frame_paths), track_anchor_strategy, num_track_anchors):
            query_ids = select_query_tokens(qwen_cache.layout, anchor, max_track_query_tokens)
            if query_ids.numel() == 0:
                continue
            query_xy = qwen_cache.layout.center_vggt_xy[query_ids]
            track_xy, track_vis, track_conf = vggt.track_from_anchor(list(record.frame_paths), anchor, query_xy)
            track_xy_rows.append(track_xy)
            track_vis_rows.append(track_vis)
            track_conf_rows.append(track_conf)
            anchor_rows.append(anchor)
            query_id_rows.append(query_ids)

        if track_xy_rows:
            max_q = max(x.shape[1] for x in track_xy_rows)
            geometry = attach_tracks(
                geometry,
                track_xy=_pad_track_tensor(track_xy_rows, max_q, fill=float("nan")),
                track_vis=_pad_track_tensor(track_vis_rows, max_q, fill=0.0),
                track_conf=_pad_track_tensor(track_conf_rows, max_q, fill=0.0),
                track_anchor_frames=torch.tensor(anchor_rows, dtype=torch.long),
                track_query_token_ids=_pad_token_ids(query_id_rows, max_q),
            )

        write_cache_metadata(
            clip_dir,
            {
                "clip_id": record.clip_id,
                "source_dataset": record.source_dataset,
                "backend": "real",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "qwen_checkpoint": qwen_checkpoint,
                "vggt_checkpoint": vggt_checkpoint,
                "vggt_source": str(vggt_source),
                "track_anchor_strategy": track_anchor_strategy,
                "num_track_anchors": num_track_anchors,
                "max_track_query_tokens": max_track_query_tokens,
            },
        )
        write_json(clip_dir / "source_record.json", record.__dict__)
        save_frame_transforms(clip_dir / "frame_transforms.json", qwen_cache.frame_transforms)
        write_json(clip_dir / "qwen_input.json", qwen_cache.qwen_input)
        save_token_layout(clip_dir / "token_layout.safetensors", qwen_cache.layout)
        save_semantic_tokens(clip_dir / "semantic_tokens.safetensors", qwen_cache.hidden)
        save_vggt_geometry(clip_dir / "geometry.safetensors", geometry)

    write_json(
        cache_root / "cache_index.json",
        {
            "backend": "real",
            "manifest": str(manifest),
            "num_clips": len(records),
            "clip_ids": [r.clip_id for r in records],
        },
    )
    print(cache_root)


def _pad_track_tensor(rows: list[torch.Tensor], max_q: int, *, fill: float) -> torch.Tensor:
    padded = []
    for value in rows:
        if value.shape[1] == max_q:
            padded.append(value)
            continue
        pad_shape = (value.shape[0], max_q - value.shape[1], *value.shape[2:])
        pad = torch.full(pad_shape, fill, dtype=value.dtype)
        padded.append(torch.cat([value.cpu(), pad], dim=1))
    return torch.stack(padded, dim=0)


def _pad_token_ids(rows: list[torch.Tensor], max_q: int) -> torch.Tensor:
    padded = []
    for value in rows:
        if value.numel() == max_q:
            padded.append(value.cpu())
            continue
        pad = torch.full((max_q - value.numel(),), -1, dtype=torch.long)
        padded.append(torch.cat([value.cpu(), pad], dim=0))
    return torch.stack(padded, dim=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, default=Path("runs/toy_cache"))
    parser.add_argument("--backend", choices=["toy", "real"], default="toy")
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--qwen-checkpoint", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--vggt-checkpoint", default="facebook/VGGT-1B")
    parser.add_argument("--vggt-source", type=Path, default=Path("third_party/vggt"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--track-anchor-strategy", choices=["first", "uniform_m", "all"], default="uniform_m")
    parser.add_argument("--num-track-anchors", type=int, default=3)
    parser.add_argument("--max-track-query-tokens", type=int, default=1024)
    args = parser.parse_args()

    if args.backend == "real":
        cache_real_manifest(
            args.manifest,
            args.cache_root,
            qwen_checkpoint=args.qwen_checkpoint,
            vggt_checkpoint=args.vggt_checkpoint,
            vggt_source=args.vggt_source,
            device=args.device,
            dtype=getattr(torch, args.dtype),
            track_anchor_strategy=args.track_anchor_strategy,
            num_track_anchors=args.num_track_anchors,
            max_track_query_tokens=args.max_track_query_tokens,
        )
        return
    cache_toy_manifest(args.manifest, args.cache_root, hidden_size=args.hidden_size)


if __name__ == "__main__":
    main()
