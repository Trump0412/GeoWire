from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from geowire.data.manifest import ClipRecord, load_manifest
from geowire.data.spatial_sft_dataset import SpatialSFTDataset
from geowire.geometry.graph_io import load_graph_npz
from geowire.geometry.vggt_cache import load_semantic_tokens, load_token_layout
from geowire.models.geowire import GeoWireTransport
from geowire.models.qwen3vl_bridge import Qwen3VLGeoWireForConditionalGeneration, graph_to_device
from geowire.models.qwen3vl_cache import extract_qwen_visual_tokens_online, ordered_image_messages
from geowire.training.checkpoints import load_torch_state, save_torch_state
from geowire.training.distributed import average_gradients, barrier, broadcast_parameters, cleanup, init_distributed
from geowire.training.pretrain_tip import tip_loss_step
from geowire.types import SparseGraph
from geowire.utils.io import write_json, write_jsonl
from geowire.utils.reproducibility import seed_everything


def load_deepspeed_config(
    path: Path,
    *,
    world_size: int,
    micro_batch_size: int,
    gradient_accumulation_steps: int,
) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    def replace_auto(value: object, resolved: int) -> object:
        return resolved if value == "auto" else value

    config["train_micro_batch_size_per_gpu"] = replace_auto(
        config.get("train_micro_batch_size_per_gpu", "auto"),
        micro_batch_size,
    )
    config["gradient_accumulation_steps"] = replace_auto(
        config.get("gradient_accumulation_steps", "auto"),
        gradient_accumulation_steps,
    )
    config["train_batch_size"] = replace_auto(
        config.get("train_batch_size", "auto"),
        micro_batch_size * gradient_accumulation_steps * max(1, world_size),
    )
    return config


def load_base_qwen(checkpoint: str, *, device: torch.device, dtype: torch.dtype):
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        checkpoint,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.eval().to(device)
    return processor, model


def apply_lora(model, *, r: int, alpha: int, dropout: float, target_modules: list[str]):
    from peft import LoraConfig, TaskType, get_peft_model

    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
    )
    return get_peft_model(model, config)


def freeze_qwen_base(model) -> None:
    for param in model.parameters():
        param.requires_grad_(False)
    if hasattr(model, "visual"):
        for param in model.visual.parameters():
            param.requires_grad_(False)


def pack_graphs(graphs: list[SparseGraph]) -> SparseGraph:
    """Pack independent per-clip graphs into one disconnected graph."""

    if not graphs:
        raise ValueError("cannot pack an empty graph list")
    dst_rows: list[torch.Tensor] = []
    src_rows: list[torch.Tensor] = []
    weight_rows: list[torch.Tensor] = []
    edge_type_rows: list[torch.Tensor] = []
    reproj_rows: list[torch.Tensor] = []
    cycle_rows: list[torch.Tensor] = []
    visibility_rows: list[torch.Tensor] = []
    confidence_rows: list[torch.Tensor] = []
    offset = 0
    for graph in graphs:
        dst_rows.append(graph.dst + offset)
        src_rows.append(graph.src + offset)
        weight_rows.append(graph.weight)
        edge_type_rows.append(graph.edge_type)
        reproj_rows.append(graph.reproj_error)
        cycle_rows.append(graph.cycle_error)
        visibility_rows.append(graph.visibility)
        confidence_rows.append(graph.confidence)
        offset += graph.num_nodes
    return SparseGraph(
        num_nodes=offset,
        dst=torch.cat(dst_rows),
        src=torch.cat(src_rows),
        weight=torch.cat(weight_rows),
        edge_type=torch.cat(edge_type_rows),
        reproj_error=torch.cat(reproj_rows),
        cycle_error=torch.cat(cycle_rows),
        visibility=torch.cat(visibility_rows),
        confidence=torch.cat(confidence_rows),
    )


def select_rank_batch(
    items,
    *,
    step_index: int,
    rank: int,
    world_size: int,
    batch_size: int,
) -> list:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    start = step_index * world_size * batch_size + rank * batch_size
    return [items[(start + local) % len(items)] for local in range(batch_size)]


def collect_image_inputs(messages_list):
    from qwen_vl_utils import process_vision_info

    images = []
    for messages in messages_list:
        image_inputs, video_inputs = process_vision_info(messages)
        if video_inputs:
            raise ValueError("GeoWire Phase 2 uses ordered images, not Qwen video mode")
        images.extend(image_inputs or [])
    return images


def build_qa_inputs(processor, records: ClipRecord | list[ClipRecord], *, device: torch.device):
    if isinstance(records, ClipRecord):
        batch_records = [records]
    else:
        batch_records = list(records)
    prompt_messages_list = [ordered_image_messages(record) for record in batch_records]
    full_messages_list = [
        [*prompt_messages, {"role": "assistant", "content": record.answer or ""}]
        for record, prompt_messages in zip(batch_records, prompt_messages_list, strict=True)
    ]
    prompt_texts = [
        processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        for messages in prompt_messages_list
    ]
    full_texts = [
        processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        for messages in full_messages_list
    ]
    image_inputs = collect_image_inputs(prompt_messages_list)
    inputs = processor(text=full_texts, images=image_inputs, videos=None, padding=True, return_tensors="pt")
    prompt_inputs = processor(text=prompt_texts, images=image_inputs, videos=None, padding=True, return_tensors="pt")
    labels = inputs["input_ids"].clone()
    prompt_attention = prompt_inputs.get("attention_mask")
    input_attention = inputs.get("attention_mask")
    if prompt_attention is None or input_attention is None:
        prompt_len = int(prompt_inputs["input_ids"].shape[-1])
        labels[:, :prompt_len] = -100
    else:
        prompt_lens = prompt_attention.sum(dim=-1).tolist()
        for row, prompt_len in enumerate(prompt_lens):
            valid = torch.nonzero(input_attention[row].bool(), as_tuple=False).flatten()
            labels[row, valid[: int(prompt_len)]] = -100
    if getattr(processor, "tokenizer", None) is not None and processor.tokenizer.pad_token_id is not None:
        labels[inputs["input_ids"] == processor.tokenizer.pad_token_id] = -100
    inputs["labels"] = labels
    return {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}


def load_phase1_adapter(geowire: GeoWireTransport, checkpoint: Path | None) -> None:
    if checkpoint is None:
        return
    state = load_torch_state(checkpoint)
    model_state = state.get("model", state)
    geowire.load_state_dict(model_state, strict=False)


def run_phase2(args: argparse.Namespace) -> dict[str, object]:
    seed_everything(args.seed)
    dist = init_distributed(args.device)
    device = dist.device
    dtype = getattr(torch, args.dtype)
    qa_dataset = SpatialSFTDataset.from_manifest(args.qa_manifest)
    tip_records = load_manifest(args.tip_manifest) if args.tip_manifest else []
    processor, base_model = load_base_qwen(args.qwen_checkpoint, device=device, dtype=dtype)
    freeze_qwen_base(base_model)
    if not args.no_lora:
        base_model = apply_lora(
            base_model,
            r=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            target_modules=args.lora_target,
        )
    hidden_size = int(base_model.config.text_config.hidden_size)
    geowire = GeoWireTransport(hidden_size=hidden_size, num_blocks=args.blocks).to(device)
    load_phase1_adapter(geowire, args.phase1_checkpoint)
    bridge = Qwen3VLGeoWireForConditionalGeneration(base_model, geowire, token_layout_builder=None).to(device)
    bridge.train()
    trainable_parameters = [p for p in bridge.parameters() if p.requires_grad]
    use_deepspeed = args.deepspeed_config is not None
    engine = None
    opt = None
    if use_deepspeed:
        import deepspeed

        ds_config = load_deepspeed_config(
            args.deepspeed_config,
            world_size=dist.world_size,
            micro_batch_size=args.train_micro_batch_size_per_gpu,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
        )
        opt = torch.optim.AdamW(trainable_parameters, lr=args.lr, weight_decay=args.weight_decay)
        engine, opt, _, _ = deepspeed.initialize(
            model=bridge,
            model_parameters=trainable_parameters,
            optimizer=opt,
            config=ds_config,
        )
        train_model = engine
    else:
        broadcast_parameters(bridge, dist, only_trainable=True)
        opt = torch.optim.AdamW(trainable_parameters, lr=args.lr, weight_decay=args.weight_decay)
        train_model = bridge

    rows: list[dict[str, object]] = []
    qa_to_tip = max(1, args.qa_to_tip)
    for step in range(args.steps):
        use_tip = bool(tip_records) and ((step + 1) % (qa_to_tip + 1) == 0)
        if not use_deepspeed:
            opt.zero_grad(set_to_none=True)
        if use_tip:
            tip_step = step // (qa_to_tip + 1)
            records = select_rank_batch(
                tip_records,
                step_index=tip_step,
                rank=dist.rank,
                world_size=dist.world_size,
                batch_size=args.train_micro_batch_size_per_gpu,
            )
            active_bridge = train_model.module if use_deepspeed else bridge
            tip_model = train_model.module.geowire if use_deepspeed else geowire
            tip_dtype = next(tip_model.parameters()).dtype
            clean_rows = []
            frame_index_rows = []
            graphs = []
            frame_offset = 0
            for record in records:
                clip_dir = args.cache_root / record.clip_id
                layout = load_token_layout(clip_dir / "token_layout.safetensors")
                graph = load_graph_npz(clip_dir / "graph_coo.npz")
                if args.tip_feature_mode == "online_qwen":
                    clean = extract_qwen_visual_tokens_online(
                        record,
                        processor=processor,
                        model=active_bridge.base_model,
                        device=device,
                        dtype=tip_dtype,
                    )
                else:
                    clean = load_semantic_tokens(clip_dir / "semantic_tokens.safetensors").to(
                        device=device,
                        dtype=tip_dtype,
                    )
                if clean.shape[0] != graph.num_nodes or clean.shape[0] != layout.frame_index.numel():
                    raise RuntimeError(
                        f"TIP token count mismatch for {record.clip_id}: "
                        f"clean={clean.shape[0]}, graph={graph.num_nodes}, layout={layout.frame_index.numel()}"
                    )
                clean_rows.append(clean)
                frame_index_rows.append(layout.frame_index.to(device) + frame_offset)
                frame_offset += int(layout.num_frames)
                graphs.append(graph)
            clean = torch.cat(clean_rows, dim=0)
            frame_index = torch.cat(frame_index_rows, dim=0)
            graph = graph_to_device(pack_graphs(graphs), device)
            loss, metrics = tip_loss_step(
                tip_model,
                clean,
                graph,
                frame_index,
                mask_ratio=args.mask_ratio,
                lambda_sub=args.lambda_sub,
                lambda_keep=args.lambda_keep,
            )
            loss = loss * args.lambda_tip_effective
            row = {
                "step": step + 1,
                "rank": dist.rank,
                "world_size": dist.world_size,
                "mode": "tip",
                "clip_id": ",".join(record.clip_id for record in records),
                "micro_batch_size": len(records),
                **metrics,
            }
        else:
            records = select_rank_batch(
                qa_dataset,
                step_index=step,
                rank=dist.rank,
                world_size=dist.world_size,
                batch_size=args.train_micro_batch_size_per_gpu,
            )
            inputs = build_qa_inputs(processor, records, device=device)
            graph = graph_to_device(
                pack_graphs([load_graph_npz(args.cache_root / record.clip_id / "graph_coo.npz") for record in records]),
                device,
            )
            outputs = train_model(graph=graph, **inputs)
            loss = outputs.loss
            row = {
                "step": step + 1,
                "rank": dist.rank,
                "world_size": dist.world_size,
                "mode": "qa",
                "clip_id": ",".join(record.clip_id for record in records),
                "micro_batch_size": len(records),
                "loss": float(loss.detach()),
            }
        if use_deepspeed:
            train_model.backward(loss)
            train_model.step()
        else:
            loss.backward()
            average_gradients(bridge, dist)
            opt.step()
        if dist.is_main:
            rows.append(row)
        if dist.is_main and ((step + 1) % args.log_every == 0 or step == args.steps - 1):
            print(row)

    barrier(dist)
    if dist.is_main:
        save_model = train_model.module if use_deepspeed else bridge
        args.output.mkdir(parents=True, exist_ok=True)
        save_torch_state(
            args.output / "phase2_adapters.pt",
            {
                "geowire": save_model.geowire.state_dict(),
                "qwen_trainable": {
                    k: v.detach().cpu() for k, v in save_model.base_model.state_dict().items() if "lora_" in k
                },
            },
        )
        write_jsonl(args.output / "metrics.jsonl", rows)
        final = {
            "steps": args.steps,
            "output": str(args.output),
            "world_size": dist.world_size,
            "final": rows[-1] if rows else {},
        }
        write_json(args.output / "metrics.json", final)
        trainer_state = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
        trainer_state["world_size"] = dist.world_size
        trainer_state["deepspeed_enabled"] = use_deepspeed
        write_json(args.output / "trainer_state.json", trainer_state)
    else:
        final = {"steps": args.steps, "rank": dist.rank, "world_size": dist.world_size}
    cleanup(dist)
    return final


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa-manifest", type=Path, required=True)
    parser.add_argument("--tip-manifest", type=Path)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--qwen-checkpoint", default="Qwen/Qwen3-VL-4B-Instruct")
    parser.add_argument("--phase1-checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=Path("runs/phase2_sft"))
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--tip-feature-mode", choices=["online_qwen", "cached"], default="online_qwen")
    parser.add_argument("--deepspeed-config", type=Path)
    parser.add_argument("--train-micro-batch-size-per-gpu", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2.0e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--qa-to-tip", type=int, default=15)
    parser.add_argument("--lambda-tip-effective", type=float, default=0.20)
    parser.add_argument("--mask-ratio", type=float, default=0.15)
    parser.add_argument("--lambda-sub", type=float, default=0.25)
    parser.add_argument("--lambda-keep", type=float, default=0.02)
    parser.add_argument("--no-lora", action="store_true")
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target",
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    args = parser.parse_args()
    result = run_phase2(args)
    if int(os.environ.get("RANK", "0")) == 0:
        print(result)
