from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DiagnosticMetrics:
    masked_token_cosine: float
    transport_retrieval_at_1: float
    transport_retrieval_at_5: float
    substitution_consistency: float
    non_edge_leakage: float
    random_graph_gap: float
    shuffled_graph_gap: float
    edge_certification_coverage: float
