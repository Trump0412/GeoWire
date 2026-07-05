from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from _bootstrap import bootstrap

bootstrap()

from geowire.geometry.qwen_layout import QwenTokenLayoutBuilder
from geowire.geometry.transforms import apply_affine, make_frame_transform


def run_check() -> dict:
    sizes = [(320, 240), (240, 320), (640, 360), (360, 640)]
    reports = []
    for i, raw_size in enumerate(sizes):
        ft = make_frame_transform(i, raw_size, (448, 448), (518, 518))
        pts = torch.tensor(
            [
                [0.5, 0.5],
                [raw_size[0] - 0.5, raw_size[1] - 0.5],
                [raw_size[0] / 2, raw_size[1] / 2],
            ],
            dtype=torch.float64,
        )
        q = apply_affine(ft.raw_to_qwen, pts)
        back = apply_affine(ft.qwen_to_raw, q)
        max_err = float((back - pts).abs().max())
        reports.append({"raw_size_wh": raw_size, "max_roundtrip_px": max_err})

    fts = tuple(make_frame_transform(i, (320, 240), (448, 448), (518, 518)) for i in range(2))
    layout = QwenTokenLayoutBuilder(hidden_size=16).build(fts, ((14, 14), (14, 14)))
    return {
        "roundtrip": reports,
        "token_count": int(layout.center_qwen_xy.shape[0]),
        "valid_tokens": int(layout.valid.sum()),
        "passed": all(r["max_roundtrip_px"] < 0.5 for r in reports),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", type=Path)
    args = parser.parse_args()
    report = run_check()
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.write:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(text + "\n", encoding="utf-8")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
