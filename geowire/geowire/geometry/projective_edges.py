from __future__ import annotations

import torch


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
