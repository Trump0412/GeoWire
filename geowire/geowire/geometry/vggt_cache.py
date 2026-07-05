from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from geowire.constants import CACHE_SCHEMA, DEFAULT_QWEN_CHECKPOINT, DEFAULT_VGGT_CHECKPOINT
from geowire.types import Affine2D, FrameTransform, TokenLayout, VGGTGeometry
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


def frame_transform_from_dict(data: dict) -> FrameTransform:
    return FrameTransform(
        frame_id=int(data["frame_id"]),
        raw_size_wh=tuple(data["raw_size_wh"]),
        qwen_size_wh=tuple(data["qwen_size_wh"]),
        vggt_size_wh=tuple(data["vggt_size_wh"]),
        raw_to_qwen=Affine2D(torch.tensor(data["raw_to_qwen"], dtype=torch.float64)),
        qwen_to_raw=Affine2D(torch.tensor(data["qwen_to_raw"], dtype=torch.float64)),
        raw_to_vggt=Affine2D(torch.tensor(data["raw_to_vggt"], dtype=torch.float64)),
        vggt_to_raw=Affine2D(torch.tensor(data["vggt_to_raw"], dtype=torch.float64)),
        valid_qwen_rect_xyxy=tuple(float(x) for x in data["valid_qwen_rect_xyxy"]),
        valid_vggt_rect_xyxy=tuple(float(x) for x in data["valid_vggt_rect_xyxy"]),
    )


def save_frame_transforms(path: str | Path, frame_transforms: tuple[FrameTransform, ...]) -> None:
    write_json(path, [frame_transform_to_dict(ft) for ft in frame_transforms])


def load_frame_transforms(path: str | Path) -> tuple[FrameTransform, ...]:
    return tuple(frame_transform_from_dict(item) for item in read_json(path))


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


def save_vggt_geometry(path: str | Path, geometry: VGGTGeometry) -> None:
    """Save frozen VGGT geometry tensors for Phase 0 graph construction."""

    tensors = {
        "extrinsic_cw": geometry.extrinsic_cw.detach().cpu().float(),
        "intrinsic": geometry.intrinsic.detach().cpu().float(),
        "depth": geometry.depth.detach().cpu().float(),
        "depth_conf": geometry.depth_conf.detach().cpu().float(),
        "world_points_head": geometry.world_points_head.detach().cpu().float(),
        "world_points_unproj": geometry.world_points_unproj.detach().cpu().float(),
        "point_conf": geometry.point_conf.detach().cpu().float(),
    }
    optional = {
        "track_xy": geometry.track_xy,
        "track_vis": geometry.track_vis,
        "track_conf": geometry.track_conf,
        "track_anchor_frames": geometry.track_anchor_frames,
        "track_query_token_ids": geometry.track_query_token_ids,
    }
    for name, value in optional.items():
        if value is not None:
            tensors[name] = value.detach().cpu()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path))


def load_vggt_geometry(cache_dir: str | Path) -> VGGTGeometry:
    cache_dir = Path(cache_dir)
    data = load_file(str(cache_dir / "geometry.safetensors"))
    frame_transforms = load_frame_transforms(cache_dir / "frame_transforms.json")
    return VGGTGeometry(
        extrinsic_cw=data["extrinsic_cw"].float(),
        intrinsic=data["intrinsic"].float(),
        depth=data["depth"].float(),
        depth_conf=data["depth_conf"].float(),
        world_points_head=data["world_points_head"].float(),
        world_points_unproj=data["world_points_unproj"].float(),
        point_conf=data["point_conf"].float(),
        frame_transforms=frame_transforms,
        track_xy=data.get("track_xy"),
        track_vis=data.get("track_vis"),
        track_conf=data.get("track_conf"),
        track_anchor_frames=data.get("track_anchor_frames"),
        track_query_token_ids=data.get("track_query_token_ids"),
    )
