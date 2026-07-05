from __future__ import annotations

import re
from dataclasses import dataclass


CHOICE_RE = re.compile(r"\b([A-D])\b", re.IGNORECASE)


@dataclass(frozen=True)
class ScoreResult:
    total: int
    correct: int
    accuracy: float


def normalize_answer(text: str, normalizer: str = "exact") -> str:
    text = str(text).strip()
    if normalizer == "spatial_choice":
        match = CHOICE_RE.search(text)
        if match:
            return match.group(1).upper()
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^[\"'`]+|[\"'`.。]+$", "", text)
    return text.strip()


def score_predictions(rows: list[dict], *, normalizer: str = "exact") -> tuple[ScoreResult, list[dict]]:
    details: list[dict] = []
    correct = 0
    for idx, row in enumerate(rows):
        prediction = normalize_answer(row.get("prediction", ""), normalizer)
        answer = normalize_answer(row.get("answer", row.get("label", "")), normalizer)
        ok = bool(answer) and prediction == answer
        correct += int(ok)
        details.append(
            {
                "index": idx,
                "id": row.get("id", row.get("clip_id", idx)),
                "prediction_norm": prediction,
                "answer_norm": answer,
                "correct": ok,
            }
        )
    total = len(rows)
    return ScoreResult(total=total, correct=correct, accuracy=(correct / total if total else 0.0)), details
