from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import yaml

from _bootstrap import bootstrap

bootstrap()

from geowire.evaluation.scoring import score_predictions
from geowire.utils.io import iter_jsonl, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/eval_vsi.yaml"))
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--write", type=Path, default=Path("runs/eval/report.json"))
    parser.add_argument("--details", type=Path)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    rows = list(iter_jsonl(args.predictions))
    normalizer = config.get("answer_normalizer_id", "exact")
    score, details = score_predictions(rows, normalizer=normalizer)
    report = {
        "config": str(args.config),
        "benchmark_name": config.get("benchmark_name"),
        "answer_normalizer_id": normalizer,
        "score": asdict(score),
    }
    write_json(args.write, report)
    if args.details:
        write_json(args.details, {"items": details})
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
