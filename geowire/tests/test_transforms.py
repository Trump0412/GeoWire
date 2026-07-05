from __future__ import annotations

import torch

from geowire.geometry.transforms import apply_affine, make_frame_transform


def test_resize_pad_roundtrip() -> None:
    for raw_size in [(320, 240), (240, 320), (640, 360), (360, 640)]:
        ft = make_frame_transform(0, raw_size, (448, 448), (518, 518))
        pts = torch.tensor(
            [[0.5, 0.5], [raw_size[0] - 0.5, raw_size[1] - 0.5], [raw_size[0] / 2, raw_size[1] / 2]],
            dtype=torch.float32,
        )
        qwen = apply_affine(ft.raw_to_qwen, pts)
        raw = apply_affine(ft.qwen_to_raw, qwen)
        assert torch.allclose(raw, pts, atol=1e-5)


def test_valid_rect_is_inside_canvas() -> None:
    ft = make_frame_transform(0, (640, 360), (448, 448), (518, 518))
    x0, y0, x1, y1 = ft.valid_qwen_rect_xyxy
    assert 0 <= x0 < x1 <= 448
    assert 0 <= y0 < y1 <= 448
