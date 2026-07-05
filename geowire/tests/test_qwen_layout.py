from __future__ import annotations

from geowire.geometry.qwen_layout import QwenTokenLayoutBuilder, ordered_image_prompt
from geowire.geometry.transforms import make_frame_transform


def test_qwen_layout_offsets() -> None:
    fts = tuple(make_frame_transform(i, (320, 240), (448, 448), (518, 518)) for i in range(2))
    layout = QwenTokenLayoutBuilder(hidden_size=32).build(fts, ((4, 5), (2, 3)))
    assert layout.token_offsets.tolist() == [0, 20, 26]
    assert layout.center_qwen_xy.shape == (26, 2)
    assert layout.center_raw_xy.shape == (26, 2)
    assert layout.center_vggt_xy.shape == (26, 2)


def test_ordered_image_prompt() -> None:
    prompt = ordered_image_prompt(2, "Where is the red box?")
    assert "[Frame 0" in prompt
    assert "<image>" in prompt
    assert "Question:" in prompt
