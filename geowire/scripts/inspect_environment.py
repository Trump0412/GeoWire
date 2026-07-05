from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


PACKAGES = [
    "torch",
    "torchvision",
    "transformers",
    "accelerate",
    "peft",
    "safetensors",
    "huggingface_hub",
    "qwen-vl-utils",
    "einops",
    "hydra-core",
    "omegaconf",
    "opencv-python-headless",
    "pillow",
    "pytest",
    "ruff",
    "mypy",
    "tensorboard",
    "wandb",
    "deepspeed",
    "flash-attn",
]


def package_versions() -> dict[str, str]:
    out = {}
    for pkg in PACKAGES:
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = "MISSING"
    return out


def cuda_report() -> dict:
    try:
        import torch

        return {
            "torch_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "device_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            if torch.cuda.is_available()
            else [],
        }
    except Exception as exc:
        return {"error": repr(exc)}


def nvidia_smi() -> str:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
        return result.stdout.strip() or result.stderr.strip()
    except Exception as exc:
        return repr(exc)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", type=Path)
    args = parser.parse_args()

    report = {
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "packages": package_versions(),
        "cuda": cuda_report(),
        "nvidia_smi": nvidia_smi(),
    }
    text = json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True)
    print(text)
    if args.write:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
