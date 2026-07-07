from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch

from _bootstrap import bootstrap

bootstrap()

from geowire.data.manifest import load_manifest
from geowire.geometry.graph_builder import EdgeCandidates, build_graph
from geowire.geometry.graph_io import load_graph_npz
from geowire.geometry.vggt_cache import load_semantic_tokens, load_token_layout
from geowire.models.geowire import GeoWireTransport
from geowire.models.qwen3vl_cache import extract_qwen_visual_tokens_online, load_qwen_processor_and_model
from geowire.training.checkpoints import save_torch_state
from geowire.training.distributed import average_gradients, barrier, broadcast_parameters, cleanup, init_distributed
from geowire.training.pretrain_tip import graph_control_metrics, run_tip_debug_step, tip_loss_step
from geowire.types import SparseGraph
from geowire.utils.io import write_json, write_jsonl
from geowire.utils.reproducibility import seed_everything


def graph_to_device(graph: SparseGraph, device: torch.device) -> SparseGraph:
    return SparseGraph(
        num_nodes=graph.num_nodes,
        dst=graph.dst.to(device),
        src=graph.src.to(device),
        weight=graph.weight.to(device),
        edge_type=graph.edge_type.to(device),
        reproj_error=graph.reproj_error.to(device),
        cycle_error=graph.cycle_error.to(device),
        visibility=graph.visibility.to(device),
        confidence=graph.confidence.to(device),
    )


def pack_graphs(graphs: list[SparseGraph]) -> SparseGraph:
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


def select_rank_batch(records, *, step: int, rank: int, world_size: int, batch_size: int) -> list:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    start = step * world_size * batch_size + rank * batch_size
    return [records[(start + local) % len(records)] for local in range(batch_size)]


def run_cached_training(args: argparse.Namespace) -> dict[str, float | int | str]:
    dist = init_distributed(args.device)
    records = load_manifest(args.manifest)
    if not records:
        raise SystemExit("manifest is empty")

    qwen_processor = None
    qwen_model = None
    dtype = getattr(torch, args.dtype)
    if args.tip_feature_mode == "online_qwen":
        qwen_processor, qwen_model = load_qwen_processor_and_model(
            args.qwen_checkpoint,
            device=dist.device,
            dtype=dtype,
        )
        for parameter in qwen_model.parameters():
            parameter.requires_grad_(False)
        first_hidden = extract_qwen_visual_tokens_online(
            records[0],
            processor=qwen_processor,
            model=qwen_model,
            device=dist.device,
            dtype=dtype,
        ).detach().cpu()
    else:
        first_hidden = load_semantic_tokens(args.cache_root / records[0].clip_id / "semantic_tokens.safetensors")
    model = GeoWireTransport(hidden_size=first_hidden.shape[-1], num_blocks=args.blocks)
    device = dist.device
    model.to(device)
    broadcast_parameters(model, dist)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    metrics_rows: list[dict[str, float | int | str]] = []
    started_at = time.monotonic()
    for step in range(args.steps):
        step_started_at = time.monotonic()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        batch_records = select_rank_batch(
            records,
            step=step,
            rank=dist.rank,
            world_size=dist.world_size,
            batch_size=args.train_micro_batch_size_per_gpu,
        )
        clean_rows = []
        frame_index_rows = []
        graphs = []
        frame_offset = 0
        for record in batch_records:
            clip_dir = args.cache_root / record.clip_id
            if args.tip_feature_mode == "online_qwen":
                clean = extract_qwen_visual_tokens_online(
                    record,
                    processor=qwen_processor,
                    model=qwen_model,
                    device=device,
                    dtype=next(model.parameters()).dtype,
                )
            else:
                clean = load_semantic_tokens(clip_dir / "semantic_tokens.safetensors").to(
                    device=device,
                    dtype=next(model.parameters()).dtype,
                )
            layout = load_token_layout(clip_dir / "token_layout.safetensors")
            graph = load_graph_npz(clip_dir / "graph_coo.npz")
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

        opt.zero_grad(set_to_none=True)
        loss, metrics = tip_loss_step(
            model,
            clean,
            graph,
            frame_index,
            mask_ratio=args.mask_ratio,
            lambda_sub=args.lambda_sub,
            lambda_keep=args.lambda_keep,
        )
        loss.backward()
        average_gradients(model, dist)
        opt.step()
        row: dict[str, float | int | str] = {
            "step": step + 1,
            "rank": dist.rank,
            "world_size": dist.world_size,
            "clip_id": ",".join(record.clip_id for record in batch_records),
            "micro_batch_size": len(batch_records),
            **metrics,
        }
        row["step_seconds"] = time.monotonic() - step_started_at
        row["elapsed_seconds"] = time.monotonic() - started_at
        if device.type == "cuda":
            row["cuda_peak_allocated_gb"] = torch.cuda.max_memory_allocated(device) / (1024**3)
            row["cuda_peak_reserved_gb"] = torch.cuda.max_memory_reserved(device) / (1024**3)
        if (step + 1) % args.eval_every == 0 or step == args.steps - 1:
            row.update(graph_control_metrics(model, clean, graph, frame_index, mask_ratio=args.mask_ratio))
        if dist.is_main:
            metrics_rows.append(row)
            if (step + 1) % args.log_every == 0 or step == args.steps - 1:
                print(row, flush=True)

    barrier(dist)
    if dist.is_main:
        args.output.mkdir(parents=True, exist_ok=True)
        save_torch_state(
            args.output / "geowire_adapter.pt",
            {
                "model": model.state_dict(),
                "hidden_size": first_hidden.shape[-1],
                "blocks": args.blocks,
            },
        )
        write_jsonl(args.output / "metrics.jsonl", metrics_rows)
        final = dict(metrics_rows[-1])
        final.update({"num_records": len(records), "checkpoint": str(args.output / "geowire_adapter.pt")})
        write_json(args.output / "metrics.json", final)
        write_json(
            args.output / "trainer_state.json",
            {
                "steps": args.steps,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "mask_ratio": args.mask_ratio,
                "lambda_sub": args.lambda_sub,
                "lambda_keep": args.lambda_keep,
                "manifest": str(args.manifest),
                "cache_root": str(args.cache_root),
                "tip_feature_mode": args.tip_feature_mode,
                "world_size": dist.world_size,
            },
        )
    else:
        final = {"steps": args.steps, "rank": dist.rank, "world_size": dist.world_size}
    cleanup(dist)
    return final


def run_debug(output: Path) -> dict[str, float]:
    output.mkdir(parents=True, exist_ok=True)
    hidden = torch.randn(8, 16, requires_grad=True)
    clean = hidden.detach().clone()
    dst = torch.tensor([4, 5, 6, 7])
    src = torch.tensor([0, 1, 2, 3])
    ones = torch.ones(4)
    graph = build_graph(
        8,
        [
            EdgeCandidates(
                dst=dst,
                src=src,
                weight=ones,
                edge_type=torch.full((4,), 2, dtype=torch.uint8),
                reproj_error=torch.zeros(4),
                cycle_error=torch.zeros(4),
                visibility=ones,
                confidence=ones,
            )
        ],
    )
    metrics = run_tip_debug_step(hidden, clean, graph)
    write_json(output / "metrics.json", metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("runs/tip_debug"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--eval-every", type=int, default=2)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2.0e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--mask-ratio", type=float, default=0.15)
    parser.add_argument("--lambda-sub", type=float, default=0.25)
    parser.add_argument("--lambda-keep", type=float, default=0.02)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tip-feature-mode", choices=["cached", "online_qwen"], default="cached")
    parser.add_argument("--qwen-checkpoint", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--train-micro-batch-size-per-gpu", type=int, default=1)
    args = parser.parse_args()
    seed_everything(3407)

    if args.manifest:
        if not args.cache_root:
            raise SystemExit("--cache-root is required with --manifest")
        metrics = run_cached_training(args)
    else:
        metrics = run_debug(args.output)
    if int(os.environ.get("RANK", "0")) == 0:
        print(metrics)


if __name__ == "__main__":
    main()
