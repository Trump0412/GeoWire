from __future__ import annotations

import torch


def uniform_frame_indices(num_total: int, num_frames: int) -> tuple[int, ...]:
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    if num_total <= 0:
        raise ValueError("num_total must be positive")
    if num_frames == 1:
        return (0,)
    idx = torch.linspace(0, num_total - 1, steps=num_frames).round().long().tolist()
    return tuple(int(i) for i in idx)
