from __future__ import annotations

import argparse
from pathlib import Path

import torch

from _bootstrap import bootstrap

bootstrap()

from geowire.data.manifest import load_manifest
from geowire.geometry.graph_builder import EdgeCandidates, build_graph
from geowire.geometry.graph_io import load_graph_npz
from geowire.geometry.vggt_cache import load_semantic_tokens, load_token_layout
from geowire.models.geowire import GeoWireTransport
from geowire.training.checkpoints import save_torch_state
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


def run_cached_training(args: argparse.Namespace) -> dict[str, float | int | str]:
    records = load_manifest(args.manifest)
    if not records:
        raise SystemExit("manifest is empty")

    first_hidden = load_semantic_tokens(args.cache_root / records[0].clip_id / "semantic_tokens.safetensors")
    model = GeoWireTransport(hidden_size=first_hidden.shape[-1], num_blocks=args.blocks)
    device = torch.device(args.device)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    metrics_rows: list[dict[str, float | int | str]] = []
    for step in range(args.steps):
        record = records[step % len(records)]
        clip_dir = args.cache_root / record.clip_id
        clean = load_semantic_tokens(clip_dir / "semantic_tokens.safetensors").to(device)
        layout = load_token_layout(clip_dir / "token_layout.safetensors")
        graph = graph_to_device(load_graph_npz(clip_dir / "graph_coo.npz"), device)
        frame_index = layout.frame_index.to(device)

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
        opt.step()
        row: dict[str, float | int | str] = {"step": step + 1, "clip_id": record.clip_id, **metrics}
        if (step + 1) % args.eval_every == 0 or step == args.steps - 1:
            row.update(graph_control_metrics(model, clean, graph, frame_index, mask_ratio=args.mask_ratio))
        metrics_rows.append(row)

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
        },
    )
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
    args = parser.parse_args()
    seed_everything(3407)

    if args.manifest:
        if not args.cache_root:
            raise SystemExit("--cache-root is required with --manifest")
        metrics = run_cached_training(args)
    else:
        metrics = run_debug(args.output)
    print(metrics)


if __name__ == "__main__":
    main()
