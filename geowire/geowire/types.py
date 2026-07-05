from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


@dataclass(frozen=True)
class Affine2D:
    """Homogeneous transform mapping xy pixel centers between two 2D coordinate systems."""

    matrix: torch.Tensor  # [3, 3], float64, source_xy -> target_xy


@dataclass(frozen=True)
class FrameTransform:
    frame_id: int
    raw_size_wh: tuple[int, int]
    qwen_size_wh: tuple[int, int]
    vggt_size_wh: tuple[int, int]
    raw_to_qwen: Affine2D
    qwen_to_raw: Affine2D
    raw_to_vggt: Affine2D
    vggt_to_raw: Affine2D
    valid_qwen_rect_xyxy: tuple[float, float, float, float]
    valid_vggt_rect_xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class TokenLayout:
    """One flattened visual-token sequence with explicit per-token geometry."""

    num_frames: int
    hidden_size: int
    token_offsets: torch.Tensor
    frame_index: torch.Tensor
    grid_row: torch.Tensor
    grid_col: torch.Tensor
    center_qwen_xy: torch.Tensor
    center_raw_xy: torch.Tensor
    center_vggt_xy: torch.Tensor
    valid: torch.Tensor


@dataclass(frozen=True)
class VGGTGeometry:
    extrinsic_cw: torch.Tensor
    intrinsic: torch.Tensor
    depth: torch.Tensor
    depth_conf: torch.Tensor
    world_points_head: torch.Tensor
    world_points_unproj: torch.Tensor
    point_conf: torch.Tensor
    frame_transforms: tuple[FrameTransform, ...]
    track_xy: torch.Tensor | None
    track_vis: torch.Tensor | None
    track_conf: torch.Tensor | None
    track_anchor_frames: torch.Tensor | None
    track_query_token_ids: torch.Tensor | None


@dataclass(frozen=True)
class SparseGraph:
    """COO graph where destination receives weighted messages from source."""

    num_nodes: int
    dst: torch.Tensor
    src: torch.Tensor
    weight: torch.Tensor
    edge_type: torch.Tensor
    reproj_error: torch.Tensor
    cycle_error: torch.Tensor
    visibility: torch.Tensor
    confidence: torch.Tensor


@dataclass(frozen=True)
class ClipRecord:
    clip_id: str
    scene_id: str
    source_dataset: str
    frame_paths: tuple[str, ...]
    frame_indices: tuple[int, ...]
    timestamps_s: tuple[float, ...]
    split: Literal["train", "val", "test"]
    question: str | None = None
    answer: str | None = None
    task_type: str | None = None
    static_view_permutation_allowed: bool = False
    cache_dir: str | None = None
