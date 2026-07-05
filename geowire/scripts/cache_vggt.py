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
    frame_transform_to_dict,
    save_semantic_tokens,
    save_token_layout,
    write_cache_metadata,
)
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
        write_json(clip_dir / "frame_transforms.json", [frame_transform_to_dict(ft) for ft in frame_transforms])
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, default=Path("runs/toy_cache"))
    parser.add_argument("--backend", choices=["toy", "real"], default="toy")
    parser.add_argument("--hidden-size", type=int, default=16)
    args = parser.parse_args()

    if args.backend == "real":
        raise SystemExit(
            "Real VGGT cache generation is gated behind pinned third_party/vggt inspection, "
            "VGGT-1B weights, and a successful coordinate-contract report. Use --backend toy "
            "for software smoke."
        )
    cache_toy_manifest(args.manifest, args.cache_root, hidden_size=args.hidden_size)


if __name__ == "__main__":
    main()
