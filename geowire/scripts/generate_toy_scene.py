from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from _bootstrap import bootstrap

bootstrap()

from geowire.utils.io import write_jsonl


def make_image(path: Path, shift: int) -> None:
    w, h = 320, 240
    img = Image.new("RGB", (w, h), (24, 24, 24))
    draw = ImageDraw.Draw(img)
    draw.rectangle((40 + shift, 60, 180 + shift, 190), fill=(48, 130, 210), outline=(255, 255, 255), width=3)
    draw.rectangle((190 - shift, 80, 280 - shift, 170), fill=(210, 90, 55), outline=(255, 255, 255), width=3)
    for x in range(0, w, 32):
        draw.line((x, 0, x, h), fill=(55, 55, 55))
    for y in range(0, h, 32):
        draw.line((0, y, w, y), fill=(55, 55, 55))
    img.save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    frame_paths = []
    for i, shift in enumerate((0, 18)):
        path = args.output / f"frame_{i:02d}.png"
        make_image(path, shift)
        frame_paths.append(str(path.resolve()))

    intrinsics = np.array([[240.0, 0.0, 160.0], [0.0, 240.0, 120.0], [0.0, 0.0, 1.0]])
    metadata = {
        "camera_intrinsics": intrinsics.tolist(),
        "note": "Synthetic two-frame scene for coordinate and sparse-graph smoke tests.",
    }
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    write_jsonl(
        args.manifest,
        [
            {
                "clip_id": "toy_scene_clip_000",
                "scene_id": "toy_scene",
                "source_dataset": "toy",
                "frame_paths": frame_paths,
                "frame_indices": [0, 1],
                "timestamps_s": [0.0, 0.5],
                "split": "train",
                "question": "Which colored rectangle moved right?",
                "answer": "blue",
                "task_type": "spatial_relation",
                "static_view_permutation_allowed": False,
            }
        ],
    )
    print(args.manifest)


if __name__ == "__main__":
    main()
