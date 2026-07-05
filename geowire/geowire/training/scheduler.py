from __future__ import annotations


def linear_warmup_decay(step: int, max_steps: int, warmup_steps: int) -> float:
    if step < warmup_steps:
        return step / max(warmup_steps, 1)
    return max(0.0, (max_steps - step) / max(max_steps - warmup_steps, 1))
