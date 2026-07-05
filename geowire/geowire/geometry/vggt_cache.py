from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from geowire.constants import CACHE_SCHEMA, DEFAULT_QWEN_CHECKPOINT, DEFAULT_VGGT_CHECKPOINT
from geowire.types import FrameTransform, TokenLayout
from geowire.utils.io import read_json
from geowire.utils.io import write_json


def cache_metadata_path(cache_dir: str | Path) -> Path:
    return Path(cache_dir) / "metadata.json"


def read_cache_metadata(cache_dir: str | Path) -> dict:
    return read_json(cache_metadata_path(cache_dir))


def write_cache_metadata(cache_dir: str | Path, metadata: dict) -> None:
    payload = {
        "cache_schema": CACHE_SCHEMA,
        "vggt_checkpoint": DEFAULT_VGGT_CHECKPOINT,
        "qwen_checkpoint": DEFAULT_QWEN_CHECKPOINT,
        **metadata,
    }
    write_json(cache_metadata_path(cache_dir), payload)


def frame_transform_to_dict(ft: FrameTransform) -> dict:
    return {
        "frame_id": ft.frame_id,
        "raw_size_wh": ft.raw_size_wh,
        "qwen_size_wh": ft.qwen_size_wh,
        "vggt_size_wh": ft.vggt_size_wh,
        "raw_to_qwen": ft.raw_to_qwen.matrix.tolist(),
        "qwen_to_raw": ft.qwen_to_raw.matrix.tolist(),
        "raw_to_vggt": ft.raw_to_vggt.matrix.tolist(),
        "vggt_to_raw": ft.vggt_to_raw.matrix.tolist(),
        "valid_qwen_rect_xyxy": ft.valid_qwen_rect_xyxy,
        "valid_vggt_rect_xyxy": ft.valid_vggt_rect_xyxy,
    }


def save_token_layout(path: str | Path, layout: TokenLayout) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "token_offsets": layout.token_offsets,
            "frame_index": layout.frame_index,
            "grid_row": layout.grid_row,
            "grid_col": layout.grid_col,
            "center_qwen_xy": layout.center_qwen_xy,
            "center_raw_xy": layout.center_raw_xy,
            "center_vggt_xy": layout.center_vggt_xy,
            "valid": layout.valid.to(torch.uint8),
            "meta": torch.tensor([layout.num_frames, layout.hidden_size], dtype=torch.long),
        },
        str(path),
    )


def load_token_layout(path: str | Path) -> TokenLayout:
    data = load_file(str(path))
    meta = data["meta"].to(torch.long)
    return TokenLayout(
        num_frames=int(meta[0]),
        hidden_size=int(meta[1]),
        token_offsets=data["token_offsets"].to(torch.long),
        frame_index=data["frame_index"].to(torch.long),
        grid_row=data["grid_row"].to(torch.long),
        grid_col=data["grid_col"].to(torch.long),
        center_qwen_xy=data["center_qwen_xy"].float(),
        center_raw_xy=data["center_raw_xy"].float(),
        center_vggt_xy=data["center_vggt_xy"].float(),
        valid=data["valid"].bool(),
    )


def save_semantic_tokens(path: str | Path, hidden: torch.Tensor) -> None:
    """Save frozen semantic visual tokens for Phase 1 TIP training."""

    if hidden.ndim != 2:
        raise ValueError("hidden must be [N, D]")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_file({"hidden": hidden.detach().cpu().float()}, str(path))


def load_semantic_tokens(path: str | Path) -> torch.Tensor:
    data = load_file(str(path))
    if "hidden" not in data:
        raise KeyError(f"{path} does not contain tensor 'hidden'")
    hidden = data["hidden"].float()
    if hidden.ndim != 2:
        raise ValueError("cached hidden tensor must be [N, D]")
    return hidden
