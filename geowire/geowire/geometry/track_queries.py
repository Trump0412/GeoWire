from __future__ import annotations

from dataclasses import dataclass

import torch

from geowire.constants import EDGE_TRACK
from geowire.geometry.certify_edges import certify_edges, geometric_weight
from geowire.geometry.graph_builder import EdgeCandidates
from geowire.geometry.transforms import apply_affine, points_in_rect
from geowire.types import TokenLayout, VGGTGeometry


@dataclass(frozen=True)
class TrackEdgeConfig:
    visibility_min: float = 0.5
    confidence_min: float = 0.5
    reproj_error_px_max: float = 8.0
    tau_reproj: float = 4.0


def nearest_valid_token_in_frame(layout: TokenLayout, frame_id: int, qwen_xy: torch.Tensor) -> tuple[int, float] | None:
    """Return nearest valid flattened token id in one frame for one Qwen xy point."""

    frame_mask = (layout.frame_index == int(frame_id)) & layout.valid
    ids = torch.nonzero(frame_mask, as_tuple=False).flatten()
    if ids.numel() == 0:
        return None
    centers = layout.center_qwen_xy[ids].to(qwen_xy.device)
    dist = torch.linalg.norm(centers - qwen_xy.to(centers.dtype), dim=-1)
    best = int(torch.argmin(dist).item())
    return int(ids[best]), float(dist[best].item())


def vggt_xy_to_qwen_xy(geometry: VGGTGeometry, frame_id: int, xy_vggt: torch.Tensor) -> torch.Tensor:
    ft = geometry.frame_transforms[int(frame_id)]
    raw_xy = apply_affine(ft.vggt_to_raw, xy_vggt)
    return apply_affine(ft.raw_to_qwen, raw_xy)


def track_edge_candidates(
    layout: TokenLayout,
    geometry: VGGTGeometry,
    *,
    config: TrackEdgeConfig | None = None,
) -> EdgeCandidates:
    """Build symmetric track edges from cached arbitrary-anchor VGGT tracks."""

    config = config or TrackEdgeConfig()
    empty_long = torch.empty(0, dtype=torch.long)
    empty_float = torch.empty(0, dtype=torch.float32)
    if (
        geometry.track_xy is None
        or geometry.track_vis is None
        or geometry.track_conf is None
        or geometry.track_anchor_frames is None
        or geometry.track_query_token_ids is None
    ):
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

    dst: list[int] = []
    src: list[int] = []
    reproj: list[float] = []
    vis_values: list[float] = []
    conf_values: list[float] = []
    for anchor_idx, anchor_frame in enumerate(geometry.track_anchor_frames.tolist()):
        query_token_ids = geometry.track_query_token_ids[anchor_idx].to(torch.long)
        for query_idx, anchor_token in enumerate(query_token_ids.tolist()):
            if anchor_token < 0 or anchor_token >= layout.frame_index.numel() or not bool(layout.valid[anchor_token]):
                continue
            for target_frame in range(layout.num_frames):
                if int(target_frame) == int(anchor_frame):
                    continue
                xy_vggt = geometry.track_xy[anchor_idx, target_frame, query_idx].float()
                ft = geometry.frame_transforms[target_frame]
                if not bool(points_in_rect(xy_vggt, ft.valid_vggt_rect_xyxy)):
                    continue
                qwen_xy = vggt_xy_to_qwen_xy(geometry, target_frame, xy_vggt)
                nearest = nearest_valid_token_in_frame(layout, target_frame, qwen_xy)
                if nearest is None:
                    continue
                target_token, dist_px = nearest
                visibility = float(geometry.track_vis[anchor_idx, target_frame, query_idx].item())
                confidence = float(geometry.track_conf[anchor_idx, target_frame, query_idx].item())
                keep = certify_edges(
                    visibility=torch.tensor([visibility]),
                    confidence=torch.tensor([confidence]),
                    reproj_error=torch.tensor([dist_px]),
                    cycle_error=None,
                    visibility_min=config.visibility_min,
                    confidence_min=config.confidence_min,
                    reproj_error_px_max=config.reproj_error_px_max,
                )
                if not bool(keep[0]):
                    continue
                dst.extend([anchor_token, target_token])
                src.extend([target_token, anchor_token])
                reproj.extend([dist_px, dist_px])
                vis_values.extend([visibility, visibility])
                conf_values.extend([confidence, confidence])

    if not dst:
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

    reproj_t = torch.tensor(reproj, dtype=torch.float32)
    vis_t = torch.tensor(vis_values, dtype=torch.float32)
    conf_t = torch.tensor(conf_values, dtype=torch.float32)
    weight = geometric_weight(
        visibility=vis_t,
        confidence=conf_t,
        reproj_error=reproj_t,
        cycle_error=None,
        tau_reproj=config.tau_reproj,
    )
    return EdgeCandidates(
        dst=torch.tensor(dst, dtype=torch.long),
        src=torch.tensor(src, dtype=torch.long),
        weight=weight,
        edge_type=torch.full((len(dst),), EDGE_TRACK, dtype=torch.uint8),
        reproj_error=reproj_t,
        cycle_error=torch.full((len(dst),), float("nan"), dtype=torch.float32),
        visibility=vis_t,
        confidence=conf_t,
    )
