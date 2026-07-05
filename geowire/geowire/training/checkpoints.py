from __future__ import annotations

from pathlib import Path

import torch


def save_torch_state(path: str | Path, state: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_torch_state(path: str | Path, *, map_location: str = "cpu") -> dict:
    return torch.load(path, map_location=map_location)
