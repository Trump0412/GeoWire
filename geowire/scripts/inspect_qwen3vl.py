from __future__ import annotations

import argparse
import json
from importlib.metadata import version
from pathlib import Path

from _bootstrap import bootstrap

bootstrap()

from geowire.models.qwen3vl_vision import require_qwen3vl_transformers_support


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="Qwen/Qwen3-VL-4B-Instruct")
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
        from transformers import AutoConfig, AutoModelForVision2Seq, AutoProcessor

        cfg = AutoConfig.from_pretrained(args.checkpoint, trust_remote_code=True)
        proc = AutoProcessor.from_pretrained(args.checkpoint, trust_remote_code=True)
        model = AutoModelForVision2Seq.from_pretrained(
            args.checkpoint,
            trust_remote_code=True,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
        report["config_class"] = cfg.__class__.__name__
        report["processor_class"] = proc.__class__.__name__
        report["model_class"] = model.__class__.__name__
        report["candidate_module_names"] = [
            name
            for name, _module in model.named_modules()
            if any(k in name.lower() for k in ("visual", "vision", "merger", "projector", "language"))
        ][:300]

    text = json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True)
    print(text)
    if args.write:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
