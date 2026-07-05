from __future__ import annotations

import torch
import torch.nn.functional as F


def mean_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a, b, dim=-1).mean().detach().cpu())


def retrieval_at_1(query: torch.Tensor, candidates: torch.Tensor, target_index: torch.Tensor) -> float:
    sim = query @ candidates.T
    pred = sim.argmax(dim=-1)
    return float((pred.cpu() == target_index.cpu()).float().mean())
