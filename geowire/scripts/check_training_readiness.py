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
from geowire.models.qwen3vl_vision import require_qwen3vl_transformers_support


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "MISSING"


def path_status(path: Path | None) -> dict[str, object]:
    if path is None:
        return {"provided": False, "exists": False}
    return {"provided": True, "path": str(path), "exists": path.exists(), "is_dir": path.is_dir()}


def check_phase1_cache(manifest: Path | None, cache_root: Path | None) -> tuple[bool, list[str], dict[str, object]]:
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
        needed = ["token_layout.safetensors", "semantic_tokens.safetensors", "graph_coo.npz", "metadata.json"]
        absent = [name for name in needed if not (clip_dir / name).exists()]
        if absent:
            missing[record.clip_id] = absent
    detail["records"] = len(records)
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["phase1", "phase2"], default="phase1")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--qwen-path", type=Path, default=Path("/mnt/guojh/lq/new/weights/base_models/Qwen3-VL-4B-Instruct"))
    parser.add_argument("--vggt-path", type=Path, default=Path("/mnt/guojh/lq/new/weights/base_models/VGGT-1B"))
    parser.add_argument("--write", type=Path)
    args = parser.parse_args()

    errors: list[str] = []
    phase1_ok, phase1_errors, phase1_detail = check_phase1_cache(args.manifest, args.cache_root)
    errors.extend(phase1_errors)
    qwen_ok, qwen_message = check_qwen3vl_support()
    if args.phase == "phase2" and not qwen_ok:
        errors.append(qwen_message)

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
