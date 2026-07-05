from __future__ import annotations

import torch

from geowire.types import Affine2D, FrameTransform


def affine_from_scale_pad(scale: float, pad_xy: tuple[float, float]) -> Affine2D:
    """Map xy pixel centers by uniform scale and pad in pixel-center coordinates."""

    mat = torch.tensor(
        [[scale, 0.0, pad_xy[0]], [0.0, scale, pad_xy[1]], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    return Affine2D(mat)


def invert_affine(affine: Affine2D) -> Affine2D:
    return Affine2D(torch.linalg.inv(affine.matrix.to(dtype=torch.float64)))


def apply_affine(affine: Affine2D, xy: torch.Tensor) -> torch.Tensor:
    """Apply a homogeneous xy transform to [..., 2] pixel-center coordinates."""

    orig_shape = xy.shape
    flat = xy.reshape(-1, 2).to(dtype=torch.float64)
    ones = torch.ones((flat.shape[0], 1), dtype=torch.float64, device=flat.device)
    homo = torch.cat([flat, ones], dim=-1)
    out = homo @ affine.matrix.to(device=flat.device).T
    return out[:, :2].reshape(orig_shape).to(dtype=xy.dtype if xy.is_floating_point() else torch.float32)


def resize_pad_affines(
    raw_size_wh: tuple[int, int],
    canvas_size_wh: tuple[int, int],
) -> tuple[Affine2D, Affine2D, tuple[float, float, float, float]]:
    """Return raw->canvas and canvas->raw transforms using aspect-preserving pad resize."""

    raw_w, raw_h = raw_size_wh
    canvas_w, canvas_h = canvas_size_wh
    if raw_w <= 0 or raw_h <= 0 or canvas_w <= 0 or canvas_h <= 0:
        raise ValueError("Image sizes must be positive")
    scale = min(canvas_w / raw_w, canvas_h / raw_h)
    resized_w = raw_w * scale
    resized_h = raw_h * scale
    pad_x = (canvas_w - resized_w) / 2.0
    pad_y = (canvas_h - resized_h) / 2.0
    raw_to_canvas = affine_from_scale_pad(scale, (pad_x, pad_y))
    canvas_to_raw = invert_affine(raw_to_canvas)
    valid = (pad_x, pad_y, pad_x + resized_w, pad_y + resized_h)
    return raw_to_canvas, canvas_to_raw, valid


def make_frame_transform(
    frame_id: int,
    raw_size_wh: tuple[int, int],
    qwen_size_wh: tuple[int, int],
    vggt_size_wh: tuple[int, int],
) -> FrameTransform:
    raw_to_qwen, qwen_to_raw, valid_qwen = resize_pad_affines(raw_size_wh, qwen_size_wh)
    raw_to_vggt, vggt_to_raw, valid_vggt = resize_pad_affines(raw_size_wh, vggt_size_wh)
    return FrameTransform(
        frame_id=frame_id,
        raw_size_wh=raw_size_wh,
        qwen_size_wh=qwen_size_wh,
        vggt_size_wh=vggt_size_wh,
        raw_to_qwen=raw_to_qwen,
        qwen_to_raw=qwen_to_raw,
        raw_to_vggt=raw_to_vggt,
        vggt_to_raw=vggt_to_raw,
        valid_qwen_rect_xyxy=valid_qwen,
        valid_vggt_rect_xyxy=valid_vggt,
    )


def points_in_rect(xy: torch.Tensor, rect_xyxy: tuple[float, float, float, float]) -> torch.Tensor:
    x0, y0, x1, y1 = rect_xyxy
    return (xy[..., 0] >= x0) & (xy[..., 0] < x1) & (xy[..., 1] >= y0) & (xy[..., 1] < y1)
