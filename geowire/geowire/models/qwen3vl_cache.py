from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from geowire.data.manifest import ClipRecord
from geowire.geometry.qwen_layout import QwenTokenLayoutBuilder
from geowire.geometry.transforms import make_frame_transform
from geowire.types import FrameTransform, TokenLayout


@dataclass(frozen=True)
class QwenVisualCache:
    hidden: torch.Tensor
    layout: TokenLayout
    frame_transforms: tuple[FrameTransform, ...]
    qwen_input: dict[str, Any]


def ordered_image_messages(record: ClipRecord) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for frame_id, frame_path in enumerate(record.frame_paths):
        label = f"Frame {frame_id}"
        if frame_id < len(record.timestamps_s):
            label += f" | timestamp {record.timestamps_s[frame_id]:.2f}s"
        content.append({"type": "text", "text": f"[{label}]"})
        content.append({"type": "image", "image": frame_path})
    question = record.question or "Describe the spatial scene across these frames."
    content.append({"type": "text", "text": f"Question: {question}"})
    return [{"role": "user", "content": content}]


def load_qwen_processor_and_model(
    checkpoint: str | Path,
    *,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
):
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(str(checkpoint), trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        str(checkpoint),
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.eval().to(device)
    return processor, model


@torch.inference_mode()
def extract_qwen_visual_cache(
    record: ClipRecord,
    *,
    processor,
    model,
    qwen_canvas_from_grid: bool = True,
) -> QwenVisualCache:
    """Extract frozen Qwen3-VL image embeddings for one ordered multi-image clip.

    The returned hidden sequence is the exact image feature sequence consumed by
    the language model's image-token replacement path.
    """

    from qwen_vl_utils import process_vision_info

    messages = ordered_image_messages(record)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    if video_inputs:
        raise ValueError("GeoWire Phase 0/1 cache uses ordered images, not Qwen video mode")
    inputs = processor(text=[text], images=image_inputs, videos=None, padding=True, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    if "pixel_values" not in inputs or "image_grid_thw" not in inputs:
        raise RuntimeError("Qwen processor did not return pixel_values and image_grid_thw")
    image_embeds, _deepstack = model.get_image_features(inputs["pixel_values"], inputs["image_grid_thw"])
    hidden = torch.cat([x.detach().cpu().float() for x in image_embeds], dim=0)

    patch_size, merge_size = _qwen_patch_and_merge(model, processor)
    grid_thw = inputs["image_grid_thw"].detach().cpu().long()
    frame_transforms = _frame_transforms_from_grid(record, grid_thw, patch_size, qwen_canvas_from_grid)
    grid_hw = tuple((int(thw[1].item() // merge_size), int(thw[2].item() // merge_size)) for thw in grid_thw)
    layout = QwenTokenLayoutBuilder(hidden_size=hidden.shape[-1]).build(frame_transforms, grid_hw)
    if layout.frame_index.numel() != hidden.shape[0]:
        raise RuntimeError(
            f"Qwen token layout mismatch: layout has {layout.frame_index.numel()} nodes, hidden has {hidden.shape[0]}"
        )
    return QwenVisualCache(
        hidden=hidden,
        layout=layout,
        frame_transforms=frame_transforms,
        qwen_input={
            "input_mode": "ordered_images",
            "image_grid_thw": grid_thw.tolist(),
            "patch_size": patch_size,
            "spatial_merge_size": merge_size,
            "prompt": text,
        },
    )


def _qwen_patch_and_merge(model, processor) -> tuple[int, int]:
    visual = getattr(model, "visual", None)
    config = getattr(model.config, "vision_config", None)
    patch = getattr(config, "patch_size", None) or getattr(getattr(processor, "image_processor", None), "patch_size", 16)
    merge = (
        getattr(visual, "spatial_merge_size", None)
        or getattr(config, "spatial_merge_size", None)
        or getattr(getattr(processor, "image_processor", None), "merge_size", 2)
    )
    return int(patch), int(merge)


def _frame_transforms_from_grid(
    record: ClipRecord,
    grid_thw: torch.Tensor,
    patch_size: int,
    qwen_canvas_from_grid: bool,
) -> tuple[FrameTransform, ...]:
    transforms = []
    for frame_id, frame_path in enumerate(record.frame_paths):
        with Image.open(frame_path) as image:
            raw_size = image.size
        if qwen_canvas_from_grid:
            h = int(grid_thw[frame_id, 1].item()) * patch_size
            w = int(grid_thw[frame_id, 2].item()) * patch_size
            qwen_size = (w, h)
        else:
            qwen_size = raw_size
        transforms.append(make_frame_transform(frame_id, raw_size, qwen_size, (518, 518)))
    return tuple(transforms)
