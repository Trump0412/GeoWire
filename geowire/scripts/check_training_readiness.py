from __future__ import annotations

import argparse
import json
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch

from _bootstrap import bootstrap

bootstrap()

from geowire.data.manifest import load_manifest
from geowire.geometry.graph_io import load_graph_npz
from geowire.geometry.vggt_cache import load_semantic_tokens, load_token_layout, read_cache_metadata
from geowire.models.qwen3vl_vision import require_qwen3vl_transformers_support
from geowire.utils.io import read_json


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "MISSING"


def path_status(path: Path | None) -> dict[str, object]:
    if path is None:
        return {"provided": False, "exists": False}
    return {"provided": True, "path": str(path), "exists": path.exists(), "is_dir": path.is_dir()}


def check_phase1_cache(
    manifest: Path | None,
    cache_root: Path | None,
    *,
    tip_feature_mode: str = "cached",
    require_real_cache: bool = False,
    min_cross_frame_coverage: float = 0.0,
) -> tuple[bool, list[str], dict[str, object]]:
    errors: list[str] = []
    detail: dict[str, object] = {"manifest": path_status(manifest), "cache_root": path_status(cache_root)}
    if manifest is None or cache_root is None:
        errors.append("phase1 requires --manifest and --cache-root")
        return False, errors, detail
    if not manifest.exists():
        errors.append(f"manifest missing: {manifest}")
        return False, errors, detail
    if not cache_root.exists():
        errors.append(f"cache root missing: {cache_root}")
        return False, errors, detail
    records = load_manifest(manifest)
    missing: dict[str, list[str]] = {}
    for record in records:
        clip_dir = cache_root / record.clip_id
        needed = ["token_layout.safetensors", "graph_coo.npz", "metadata.json"]
        if tip_feature_mode == "cached":
            needed.append("semantic_tokens.safetensors")
        if require_real_cache:
            needed.append("geometry.safetensors")
        absent = [name for name in needed if not (clip_dir / name).exists()]
        if absent:
            missing[record.clip_id] = absent
            continue
        try:
            metadata = read_cache_metadata(clip_dir)
            layout = load_token_layout(clip_dir / "token_layout.safetensors")
            graph = load_graph_npz(clip_dir / "graph_coo.npz")
            hidden = load_semantic_tokens(clip_dir / "semantic_tokens.safetensors") if tip_feature_mode == "cached" else None
        except Exception as exc:  # noqa: BLE001 - readiness should report all cache failures
            missing[record.clip_id] = [f"invalid cache: {exc}"]
            continue
        if require_real_cache and metadata.get("backend") != "real":
            missing[record.clip_id] = [f"backend is {metadata.get('backend')!r}, expected 'real'"]
            continue
        if hidden is not None and hidden.shape[0] != layout.frame_index.numel():
            missing[record.clip_id] = [f"semantic token count {hidden.shape[0]} != layout nodes {layout.frame_index.numel()}"]
            continue
        if hidden is not None and hidden.shape[-1] != layout.hidden_size:
            missing[record.clip_id] = [f"semantic hidden size {hidden.shape[-1]} != layout hidden size {layout.hidden_size}"]
            continue
        if graph.num_nodes != layout.frame_index.numel():
            missing[record.clip_id] = [f"graph nodes {graph.num_nodes} != layout nodes {layout.frame_index.numel()}"]
            continue
        row_sum = torch.zeros(graph.num_nodes, dtype=torch.float32)
        row_sum.index_add_(0, graph.dst, graph.weight.float())
        if not torch.allclose(row_sum, torch.ones_like(row_sum), atol=1e-4):
            missing[record.clip_id] = ["graph row weights are not normalized"]
            continue
        cross = layout.frame_index[graph.dst] != layout.frame_index[graph.src]
        coverage = float(torch.unique(graph.dst[cross]).numel() / max(1, graph.num_nodes))
        if coverage < min_cross_frame_coverage:
            missing[record.clip_id] = [f"cross-frame coverage {coverage:.4f} < {min_cross_frame_coverage:.4f}"]
    detail["records"] = len(records)
    detail["tip_feature_mode"] = tip_feature_mode
    detail["missing_cache_files"] = missing
    if missing:
        errors.append(f"{len(missing)} clips are missing required cache files")
    return not errors, errors, detail


def check_qwen3vl_support() -> tuple[bool, str]:
    transformers_version = package_version("transformers")
    if transformers_version == "MISSING":
        return False, "transformers is missing"
    try:
        require_qwen3vl_transformers_support(transformers_version)
    except RuntimeError as exc:
        return False, str(exc)
    return True, f"transformers {transformers_version}"


def check_parity_report(path: Path | None) -> tuple[bool, str, dict[str, object]]:
    if path is None:
        return False, "phase2 requires --parity-report", {"provided": False}
    status = path_status(path)
    if not path.exists():
        return False, f"parity report missing: {path}", status
    report = read_json(path)
    passed = bool(report.get("passed"))
    if not passed:
        return False, f"parity report did not pass: {path}", {"provided": True, "path": str(path), "report": report}
    return True, f"parity report passed: {path}", {"provided": True, "path": str(path), "report": report}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["phase1", "phase2"], default="phase1")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--qwen-path", type=Path, default=Path("models/Qwen3-VL-2B-Instruct"))
    parser.add_argument("--vggt-path", type=Path, default=Path("models/VGGT-1B"))
    parser.add_argument("--require-real-cache", action="store_true")
    parser.add_argument("--tip-feature-mode", choices=["cached", "online_qwen"], default="cached")
    parser.add_argument("--min-cross-frame-coverage", type=float, default=0.0)
    parser.add_argument("--parity-report", type=Path)
    parser.add_argument("--write", type=Path)
    args = parser.parse_args()

    errors: list[str] = []
    require_real_cache = args.require_real_cache or (args.phase == "phase2" and args.tip_feature_mode == "cached")
    phase1_ok, phase1_errors, phase1_detail = check_phase1_cache(
        args.manifest,
        args.cache_root,
        tip_feature_mode=args.tip_feature_mode,
        require_real_cache=require_real_cache,
        min_cross_frame_coverage=args.min_cross_frame_coverage,
    )
    errors.extend(phase1_errors)
    qwen_ok, qwen_message = check_qwen3vl_support()
    if args.phase == "phase2" and not qwen_ok:
        errors.append(qwen_message)
    parity_ok, parity_message, parity_detail = (True, "not required", {"provided": False})
    if args.phase == "phase2":
        parity_ok, parity_message, parity_detail = check_parity_report(args.parity_report)
        if not parity_ok:
            errors.append(parity_message)

    report = {
        "phase": args.phase,
        "passed": not errors,
        "errors": errors,
        "python": sys.executable,
        "torch": package_version("torch"),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "packages": {
            "transformers": package_version("transformers"),
            "accelerate": package_version("accelerate"),
            "peft": package_version("peft"),
            "deepspeed": package_version("deepspeed"),
            "safetensors": package_version("safetensors"),
        },
        "resources": {
            "qwen": path_status(args.qwen_path),
            "vggt": path_status(args.vggt_path),
        },
        "phase1_cache": phase1_detail,
        "qwen3vl_support": {"passed": qwen_ok, "message": qwen_message},
        "parity": {"passed": parity_ok, "message": parity_message, **parity_detail},
        "require_real_cache": require_real_cache,
        "tip_feature_mode": args.tip_feature_mode,
    }
    text = json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True)
    print(text)
    if args.write:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(text + "\n", encoding="utf-8")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
