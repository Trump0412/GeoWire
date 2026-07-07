from __future__ import annotations

from dataclasses import dataclass
import os
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
    image_max_pixels = os.environ.get("GEOWIRE_IMAGE_MAX_PIXELS")
    image_min_pixels = os.environ.get("GEOWIRE_IMAGE_MIN_PIXELS")
    for frame_id, frame_path in enumerate(record.frame_paths):
        label = f"Frame {frame_id}"
        if frame_id < len(record.timestamps_s):
            label += f" | timestamp {record.timestamps_s[frame_id]:.2f}s"
        content.append({"type": "text", "text": f"[{label}]"})
        image_item: dict[str, Any] = {"type": "image", "image": frame_path}
        if image_max_pixels:
            image_item["max_pixels"] = int(image_max_pixels)
        if image_min_pixels:
            image_item["min_pixels"] = int(image_min_pixels)
        content.append(image_item)
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


@torch.inference_mode()
def extract_qwen_visual_tokens_online(
    record: ClipRecord,
    *,
    processor,
    model,
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Return frozen Qwen image features on the training device.

    This is the training-time equivalent of `extract_qwen_visual_cache`, but it
    avoids writing or reading large semantic token tensors from shared storage.
    """

    from qwen_vl_utils import process_vision_info

    messages = ordered_image_messages(record)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    if video_inputs:
        raise ValueError("GeoWire online TIP uses ordered images, not Qwen video mode")
    inputs = processor(text=[text], images=image_inputs, videos=None, padding=True, return_tensors="pt")
    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    if "pixel_values" not in inputs or "image_grid_thw" not in inputs:
        raise RuntimeError("Qwen processor did not return pixel_values and image_grid_thw")
    getter = getattr(model, "get_image_features", None)
    if getter is None and hasattr(model, "model"):
        getter = getattr(model.model, "get_image_features", None)
    if getter is None:
        raise TypeError("Qwen model does not expose get_image_features")
    image_embeds, _deepstack = getter(inputs["pixel_values"], inputs["image_grid_thw"])
    hidden = torch.cat([x.detach() for x in image_embeds], dim=0)
    if dtype is not None:
        hidden = hidden.to(dtype=dtype)
    return hidden


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
