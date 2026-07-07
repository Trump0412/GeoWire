from __future__ import annotations

import argparse
import inspect
import json
from importlib.metadata import version
from pathlib import Path

from _bootstrap import bootstrap

bootstrap()

from geowire.models.qwen3vl_vision import require_qwen3vl_transformers_support


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="Qwen/Qwen3-VL-2B-Instruct")
    parser.add_argument("--write", type=Path)
    parser.add_argument("--load-model", action="store_true")
    args = parser.parse_args()

    transformers_version = version("transformers")
    report: dict[str, object] = {
        "checkpoint": args.checkpoint,
        "transformers_version": transformers_version,
        "load_model": args.load_model,
    }
    try:
        require_qwen3vl_transformers_support(transformers_version)
        report["transformers_support"] = "ok"
    except RuntimeError as exc:
        report["transformers_support"] = f"failed: {exc}"

    if args.load_model:
        from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

        cfg = AutoConfig.from_pretrained(args.checkpoint, trust_remote_code=True)
        proc = AutoProcessor.from_pretrained(args.checkpoint, trust_remote_code=True)
        model = AutoModelForImageTextToText.from_pretrained(
            args.checkpoint,
            trust_remote_code=True,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
        report["config_class"] = cfg.__class__.__name__
        report["processor_class"] = proc.__class__.__name__
        report["model_class"] = model.__class__.__name__
        report["model_source"] = inspect.getsourcefile(model.__class__)
        report["vision_module_class"] = model.visual.__class__.__name__ if hasattr(model, "visual") else None
        report["spatial_merge_size"] = getattr(getattr(model, "visual", None), "spatial_merge_size", None)
        candidates = {}
        for name, module in model.named_modules():
            if any(k in name.lower() for k in ("visual", "vision", "merger", "projector", "language")):
                try:
                    sig = str(inspect.signature(module.forward))
                except (TypeError, ValueError):
                    sig = "<unavailable>"
                candidates[name] = f"{module.__class__.__name__}{sig}"
                if len(candidates) >= 300:
                    break
        report["candidate_modules"] = candidates

    text = json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True)
    print(text)
    if args.write:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
