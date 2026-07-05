from __future__ import annotations

import argparse
from pathlib import Path

import torch

from _bootstrap import bootstrap

bootstrap()

from geowire.data.manifest import load_manifest
from geowire.geometry.graph_builder import self_loop_graph
from geowire.geometry.graph_io import load_graph_npz
from geowire.models.geowire import GeoWireTransport
from geowire.models.qwen3vl_bridge import Qwen3VLGeoWireForConditionalGeneration, graph_to_device
from geowire.models.qwen3vl_cache import ordered_image_messages
from geowire.utils.io import write_json
from geowire.utils.reproducibility import seed_everything


def build_inputs(processor, record, *, device: torch.device):
    from qwen_vl_utils import process_vision_info

    messages = ordered_image_messages(record)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    if video_inputs:
        raise ValueError("GeoWire parity uses ordered images, not video mode")
    inputs = processor(text=[text], images=image_inputs, videos=None, padding=True, return_tensors="pt")
    return {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--clip-index", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--tolerance", type=float, default=2e-3)
    parser.add_argument("--write", type=Path, default=Path("runs/qwen_bridge_parity/report.json"))
    args = parser.parse_args()

    seed_everything(3407)
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(args.checkpoint, trust_remote_code=True)
    base_model = AutoModelForImageTextToText.from_pretrained(
        args.checkpoint,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).eval().to(device)
    records = load_manifest(args.manifest)
    record = records[args.clip_index]
    inputs = build_inputs(processor, record, device=device)

    with torch.inference_mode():
        base_out = base_model(**inputs, logits_to_keep=1)
        image_features, _ = base_model.get_image_features(inputs["pixel_values"], inputs["image_grid_thw"])
        num_nodes = int(sum(x.shape[0] for x in image_features))
        if args.cache_root and (args.cache_root / record.clip_id / "graph_coo.npz").exists():
            graph = load_graph_npz(args.cache_root / record.clip_id / "graph_coo.npz")
        else:
            graph = self_loop_graph(num_nodes)
        geowire = GeoWireTransport(hidden_size=int(base_model.config.text_config.hidden_size), num_blocks=2).to(device)
        bridge = Qwen3VLGeoWireForConditionalGeneration(base_model, geowire, token_layout_builder=None).eval()
        bridge_out = bridge(graph=graph_to_device(graph, device), **inputs, logits_to_keep=1)
        max_abs = float((base_out.logits.float() - bridge_out.logits.float()).abs().max().item())
    report = {
        "passed": max_abs <= args.tolerance,
        "max_abs_logit_diff": max_abs,
        "tolerance": args.tolerance,
        "checkpoint": args.checkpoint,
        "clip_id": record.clip_id,
        "num_visual_nodes": num_nodes,
        "graph_nodes": int(graph.num_nodes),
    }
    print(report)
    write_json(args.write, report)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
