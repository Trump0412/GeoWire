from __future__ import annotations

import argparse
from pathlib import Path

import torch

from _bootstrap import bootstrap

bootstrap()

from geowire.constants import EDGE_PROJECTIVE, EDGE_TRACK
from geowire.data.manifest import load_manifest
from geowire.geometry.graph_builder import EdgeCandidates, build_graph
from geowire.geometry.graph_io import save_graph_npz
from geowire.geometry.graph_stats import graph_stats
from geowire.geometry.projective_edges import projective_edge_candidates
from geowire.geometry.track_queries import TrackEdgeConfig, track_edge_candidates
from geowire.geometry.vggt_cache import load_token_layout, load_vggt_geometry, read_cache_metadata
from geowire.utils.io import write_json


def toy_layout_edges(cache_dir: Path) -> tuple[int, EdgeCandidates, torch.Tensor]:
    layout = load_token_layout(cache_dir / "token_layout.safetensors")
    dst: list[int] = []
    src: list[int] = []
    for frame in range(layout.num_frames - 1):
        a0 = int(layout.token_offsets[frame])
        a1 = int(layout.token_offsets[frame + 1])
        b0 = int(layout.token_offsets[frame + 1])
        b1 = int(layout.token_offsets[frame + 2])
        n = min(a1 - a0, b1 - b0)
        for local in range(n):
            i = a0 + local
            j = b0 + local
            if bool(layout.valid[i]) and bool(layout.valid[j]):
                dst.extend([i, j])
                src.extend([j, i])
    if not dst:
        empty = torch.empty(0, dtype=torch.long)
        return (
            int(layout.frame_index.numel()),
            EdgeCandidates(
                dst=empty,
                src=empty,
                weight=torch.empty(0),
                edge_type=torch.empty(0, dtype=torch.uint8),
                reproj_error=torch.empty(0),
                cycle_error=torch.empty(0),
                visibility=torch.empty(0),
                confidence=torch.empty(0),
            ),
            layout.frame_index,
        )
    num_edges = len(dst)
    return (
        int(layout.frame_index.numel()),
        EdgeCandidates(
            dst=torch.tensor(dst, dtype=torch.long),
            src=torch.tensor(src, dtype=torch.long),
            weight=torch.ones(num_edges),
            edge_type=torch.full((num_edges,), EDGE_PROJECTIVE, dtype=torch.uint8),
            reproj_error=torch.zeros(num_edges),
            cycle_error=torch.zeros(num_edges),
            visibility=torch.ones(num_edges),
            confidence=torch.ones(num_edges),
        ),
        layout.frame_index,
    )


def build_real_or_toy_edges(args: argparse.Namespace, cache_dir: Path) -> tuple[int, list[EdgeCandidates], torch.Tensor, str]:
    layout = load_token_layout(cache_dir / "token_layout.safetensors")
    metadata = read_cache_metadata(cache_dir)
    if (cache_dir / "geometry.safetensors").exists() and metadata.get("backend") == "real":
        geometry = load_vggt_geometry(cache_dir)
        candidates = [
            track_edge_candidates(
                layout,
                geometry,
                config=TrackEdgeConfig(
                    visibility_min=args.visibility_min,
                    confidence_min=args.track_conf_min,
                    reproj_error_px_max=args.reproj_error_px_max,
                    tau_reproj=args.tau_reproj,
                ),
            ),
            projective_edge_candidates(
                layout,
                geometry,
                depth_conf_min=args.depth_conf_min,
                depth_rel_error_max=args.depth_rel_error_max,
                reproj_error_px_max=args.reproj_error_px_max,
                tau_depth=args.tau_depth,
            ),
        ]
        return int(layout.frame_index.numel()), candidates, layout.frame_index, "real"
    num_nodes, toy_candidates, frame_index = toy_layout_edges(cache_dir)
    return num_nodes, [toy_candidates], frame_index, "toy"


def build_from_manifest(args: argparse.Namespace, manifest: Path, cache_root: Path, output: Path, *, topk: int) -> None:
    records = load_manifest(manifest)
    summaries: dict[str, dict[str, float | int]] = {}
    for record in records:
        clip_cache = cache_root / record.clip_id
        num_nodes, candidates, frame_index, backend = build_real_or_toy_edges(args, clip_cache)
        graph = build_graph(
            num_nodes,
            candidates,
            frame_index=frame_index,
            topk_cross_frame_edges=topk,
        )
        save_graph_npz(clip_cache / "graph_coo.npz", graph)
        summary = graph_stats(graph)
        summary["graph_backend"] = backend
        summaries[record.clip_id] = summary
        write_json(clip_cache / "debug" / "graph_stats.json", summary)
    write_json(output / "graph_summary.json", summaries)
    print(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("runs/phase0_graph"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--topk-cross-frame", type=int, default=4)
    parser.add_argument("--visibility-min", type=float, default=0.5)
    parser.add_argument("--track-conf-min", type=float, default=0.5)
    parser.add_argument("--depth-conf-min", type=float, default=1.0)
    parser.add_argument("--reproj-error-px-max", type=float, default=8.0)
    parser.add_argument("--depth-rel-error-max", type=float, default=0.08)
    parser.add_argument("--tau-reproj", type=float, default=4.0)
    parser.add_argument("--tau-depth", type=float, default=0.08)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    if args.manifest:
        if not args.cache_root:
            raise SystemExit("--cache-root is required with --manifest")
        build_from_manifest(args, args.manifest, args.cache_root, args.output, topk=args.topk_cross_frame)
        return

    num_nodes = 8
    dst = torch.tensor([4, 5, 6, 7])
    src = torch.tensor([0, 1, 2, 3])
    ones = torch.ones(4)
    candidates = EdgeCandidates(
        dst=dst,
        src=src,
        weight=ones,
        edge_type=torch.full((4,), EDGE_TRACK, dtype=torch.uint8),
        reproj_error=torch.zeros(4),
        cycle_error=torch.zeros(4),
        visibility=ones,
        confidence=ones,
    )
    graph = build_graph(num_nodes, [candidates])
    save_graph_npz(args.output / "graph_coo.npz", graph)
    write_json(args.output / "graph_stats.json", graph_stats(graph))
    print(args.output)


if __name__ == "__main__":
    main()
