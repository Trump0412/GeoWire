from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class FrameSamplingProtocol:
    num_frames: int
    strategy: Literal["uniform", "centered_uniform", "benchmark_official"]
    include_first_last: bool
    max_frames: int


@dataclass(frozen=True)
class EvaluationProtocol:
    benchmark_name: str
    sampling: FrameSamplingProtocol
    prompt_template_id: str
    answer_normalizer_id: str
    model_generation_kwargs: dict
