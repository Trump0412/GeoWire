from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from geowire.evaluation.scoring import normalize_answer, score_predictions


def test_spatial_choice_normalizer() -> None:
    assert normalize_answer("The answer is (b).", "spatial_choice") == "B"
    assert normalize_answer(" left ", "exact") == "left"


def test_score_predictions() -> None:
    score, details = score_predictions(
        [{"prediction": "A", "answer": "A"}, {"prediction": "B", "answer": "C"}],
        normalizer="spatial_choice",
    )
    assert score.total == 2
    assert score.correct == 1
    assert score.accuracy == 0.5
    assert details[0]["correct"] is True


def test_evaluate_cli(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    preds = tmp_path / "preds.jsonl"
    report = tmp_path / "report.json"
    preds.write_text(
        "\n".join(
            [
                json.dumps({"id": "a", "prediction": "A", "answer": "A"}),
                json.dumps({"id": "b", "prediction": "B", "answer": "C"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/evaluate.py",
            "--config",
            "configs/eval_vsi.yaml",
            "--predictions",
            str(preds),
            "--write",
            str(report),
        ],
        cwd=root,
        check=True,
    )
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["score"]["accuracy"] == 0.5
