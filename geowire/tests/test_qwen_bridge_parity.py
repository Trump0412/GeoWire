from __future__ import annotations

import pytest


@pytest.mark.skip(reason="Gated until Qwen3-VL source inspection and local checkpoint are available.")
def test_qwen_bridge_alpha_zero_parity() -> None:
    raise AssertionError("parity test must be implemented before Phase 2")
