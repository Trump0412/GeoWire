from __future__ import annotations

import torch

from geowire.constants import EDGE_PROJECTIVE
from geowire.geometry.certify_edges import certify_edges, geometric_weight
from geowire.geometry.graph_builder import EdgeCandidates
from geowire.geometry.track_queries import nearest_valid_token_in_frame, vggt_xy_to_qwen_xy
from geowire.geometry.transforms import points_in_rect
from geowire.types import TokenLayout, VGGTGeometry


def project_world_points(
    world_points: torch.Tensor,
    extrinsic_cw: torch.Tensor,
    intrinsic: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project world xyz points to image xy using camera-from-world OpenCV extrinsic."""

    if world_points.shape[-1] != 3:
        raise ValueError("world_points must have shape [..., 3]")
    flat = world_points.reshape(-1, 3).to(dtype=torch.float64)
    ones = torch.ones((flat.shape[0], 1), dtype=torch.float64, device=flat.device)
    world_h = torch.cat([flat, ones], dim=-1)
    cam_h = world_h @ extrinsic_cw.to(dtype=torch.float64, device=flat.device).T
    cam = cam_h[:, :3]
    z = cam[:, 2]
    valid = z > eps
    pix_h = cam @ intrinsic.to(dtype=torch.float64, device=flat.device).T
    xy = pix_h[:, :2] / z.clamp_min(eps).unsqueeze(-1)
    return xy.reshape(world_points.shape[:-1] + (2,)).to(dtype=torch.float32), valid.reshape(
        world_points.shape[:-1]
    )


def bilinear_depth_sample(depth: torch.Tensor, xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a single-channel depth map at xy pixel centers with bilinear interpolation."""

    if depth.ndim != 2:
        raise ValueError("depth must be [H, W]")
    h, w = depth.shape
    x = xy[..., 0]
    y = xy[..., 1]
    valid = (x >= 0) & (x <= w - 1) & (y >= 0) & (y <= h - 1)
    x0 = x.floor().clamp(0, w - 1).long()
    y0 = y.floor().clamp(0, h - 1).long()
    x1 = (x0 + 1).clamp(0, w - 1)
    y1 = (y0 + 1).clamp(0, h - 1)
    dx = (x - x0.float()).clamp(0, 1)
    dy = (y - y0.float()).clamp(0, 1)
    v00 = depth[y0, x0]
    v10 = depth[y0, x1]
    v01 = depth[y1, x0]
    v11 = depth[y1, x1]
    sample = (1 - dx) * (1 - dy) * v00 + dx * (1 - dy) * v10 + (1 - dx) * dy * v01 + dx * dy * v11
    return sample, valid


def bilinear_tensor_sample(image: torch.Tensor, xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a `[H, W, C]` tensor at xy pixel centers with bilinear interpolation."""

    if image.ndim != 3:
        raise ValueError("image must be [H, W, C]")
    channels = []
    valid = None
    for c in range(image.shape[-1]):
        sample, channel_valid = bilinear_depth_sample(image[..., c], xy)
        channels.append(sample)
        valid = channel_valid if valid is None else (valid & channel_valid)
    return torch.stack(channels, dim=-1), valid if valid is not None else torch.zeros(xy.shape[:-1], dtype=torch.bool)


def projective_edge_candidates(
    layout: TokenLayout,
    geometry: VGGTGeometry,
    *,
    depth_conf_min: float = 1.0,
    depth_rel_error_max: float = 0.08,
    reproj_error_px_max: float = 8.0,
    tau_depth: float = 0.08,
) -> EdgeCandidates:
    """Build directed projective edges from cached VGGT cameras, depth and unprojected points."""

    dst: list[int] = []
    src: list[int] = []
    reproj: list[float] = []
    depth_errors: list[float] = []
    confidence_values: list[float] = []

    for src_node in torch.nonzero(layout.valid, as_tuple=False).flatten().tolist():
        src_frame = int(layout.frame_index[src_node])
        src_xy = layout.center_vggt_xy[src_node].float()
        point, point_valid = bilinear_tensor_sample(geometry.world_points_unproj[src_frame], src_xy)
        if not bool(point_valid):
            continue
        point_conf, _ = bilinear_depth_sample(geometry.point_conf[src_frame], src_xy)
        for dst_frame in range(layout.num_frames):
            if dst_frame == src_frame:
                continue
            xy_proj, front = project_world_points(point, geometry.extrinsic_cw[dst_frame], geometry.intrinsic[dst_frame])
            if not bool(front):
                continue
            ft = geometry.frame_transforms[dst_frame]
            if not bool(points_in_rect(xy_proj, ft.valid_vggt_rect_xyxy)):
                continue
            z_cam = _camera_z(point, geometry.extrinsic_cw[dst_frame])
            depth_sample, depth_valid = bilinear_depth_sample(geometry.depth[dst_frame], xy_proj)
            depth_conf, _ = bilinear_depth_sample(geometry.depth_conf[dst_frame], xy_proj)
            if not bool(depth_valid):
                continue
            depth_rel = float((depth_sample - z_cam).abs().div(z_cam.abs().clamp_min(1e-6)).item())
            qwen_xy = vggt_xy_to_qwen_xy(geometry, dst_frame, xy_proj)
            nearest = nearest_valid_token_in_frame(layout, dst_frame, qwen_xy)
            if nearest is None:
                continue
            dst_node, dist_px = nearest
            confidence = float(torch.minimum(point_conf.float(), depth_conf.float()).item())
            keep = certify_edges(
                visibility=torch.ones(1),
                confidence=torch.tensor([confidence]),
                reproj_error=torch.tensor([dist_px]),
                cycle_error=None,
                confidence_min=depth_conf_min,
                reproj_error_px_max=reproj_error_px_max,
            )
            if not bool(keep[0]) or depth_rel > depth_rel_error_max:
                continue
            dst.append(dst_node)
            src.append(src_node)
            reproj.append(dist_px)
            depth_errors.append(depth_rel)
            confidence_values.append(confidence)

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

    reproj_t = torch.tensor(reproj, dtype=torch.float32)
    depth_t = torch.tensor(depth_errors, dtype=torch.float32)
    conf_t = torch.tensor(confidence_values, dtype=torch.float32)
    weight = conf_t * torch.exp(-depth_t / tau_depth) * geometric_weight(
        visibility=torch.ones_like(conf_t),
        confidence=torch.ones_like(conf_t),
        reproj_error=reproj_t,
        cycle_error=None,
    )
    return EdgeCandidates(
        dst=torch.tensor(dst, dtype=torch.long),
        src=torch.tensor(src, dtype=torch.long),
        weight=weight,
        edge_type=torch.full((len(dst),), EDGE_PROJECTIVE, dtype=torch.uint8),
        reproj_error=reproj_t,
        cycle_error=torch.full((len(dst),), float("nan"), dtype=torch.float32),
        visibility=torch.ones(len(dst), dtype=torch.float32),
        confidence=conf_t,
    )


def _camera_z(world_point: torch.Tensor, extrinsic_cw: torch.Tensor) -> torch.Tensor:
    point = world_point.reshape(1, 3).to(dtype=torch.float64, device=extrinsic_cw.device)
    ones = torch.ones((1, 1), dtype=torch.float64, device=point.device)
    world_h = torch.cat([point, ones], dim=-1)
    cam_h = world_h @ extrinsic_cw.to(dtype=torch.float64).T
    return cam_h[0, 2].to(dtype=torch.float32)
