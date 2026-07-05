from __future__ import annotations

from dataclasses import dataclass

import torch

from geowire.geometry.transforms import apply_affine, points_in_rect
from geowire.types import FrameTransform, TokenLayout


@dataclass(frozen=True)
class QwenTokenLayoutBuilder:
    """Build an explicit per-frame token layout for ordered multi-image inputs."""

    hidden_size: int

    def build(
        self,
        frame_transforms: tuple[FrameTransform, ...],
        grid_hw: tuple[tuple[int, int], ...],
    ) -> TokenLayout:
        if len(frame_transforms) != len(grid_hw):
            raise ValueError("frame_transforms and grid_hw length mismatch")

        frame_index: list[torch.Tensor] = []
        grid_row: list[torch.Tensor] = []
        grid_col: list[torch.Tensor] = []
        qwen_xy: list[torch.Tensor] = []
        raw_xy: list[torch.Tensor] = []
        vggt_xy: list[torch.Tensor] = []
        valid: list[torch.Tensor] = []
        offsets = [0]

        for frame_id, (ft, (rows, cols)) in enumerate(zip(frame_transforms, grid_hw, strict=True)):
            if rows <= 0 or cols <= 0:
                raise ValueError("Token grid rows/cols must be positive")
            qwen_w, qwen_h = ft.qwen_size_wh
            y = (torch.arange(rows, dtype=torch.float32) + 0.5) * (qwen_h / rows)
            x = (torch.arange(cols, dtype=torch.float32) + 0.5) * (qwen_w / cols)
            yy, xx = torch.meshgrid(y, x, indexing="ij")
            xy = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
            raw = apply_affine(ft.qwen_to_raw, xy)
            vggt = apply_affine(ft.raw_to_vggt, raw)

            n = rows * cols
            frame_index.append(torch.full((n,), frame_id, dtype=torch.long))
            rr, cc = torch.meshgrid(torch.arange(rows), torch.arange(cols), indexing="ij")
            grid_row.append(rr.reshape(-1).to(torch.long))
            grid_col.append(cc.reshape(-1).to(torch.long))
            qwen_xy.append(xy)
            raw_xy.append(raw)
            vggt_xy.append(vggt)
            valid.append(points_in_rect(xy, ft.valid_qwen_rect_xyxy))
            offsets.append(offsets[-1] + n)

        return TokenLayout(
            num_frames=len(frame_transforms),
            hidden_size=self.hidden_size,
            token_offsets=torch.tensor(offsets, dtype=torch.long),
            frame_index=torch.cat(frame_index),
            grid_row=torch.cat(grid_row),
            grid_col=torch.cat(grid_col),
            center_qwen_xy=torch.cat(qwen_xy),
            center_raw_xy=torch.cat(raw_xy),
            center_vggt_xy=torch.cat(vggt_xy),
            valid=torch.cat(valid),
        )


def ordered_image_prompt(num_frames: int, question: str, seconds_per_frame: float = 0.5) -> str:
    lines = []
    for i in range(num_frames):
        lines.append(f"[Frame {i} | timestamp {i * seconds_per_frame:.2f}s] <image>")
    lines.append(f"Question: {question}")
    return "\n".join(lines)
