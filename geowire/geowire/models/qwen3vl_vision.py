from __future__ import annotations


def require_qwen3vl_transformers_support(transformers_version: str) -> None:
    parts = tuple(int(p) for p in transformers_version.split(".")[:2])
    if parts < (4, 57):
        raise RuntimeError(
            f"Transformers {transformers_version} is likely too old for Qwen3-VL. "
            "Install the Qwen3-VL-compatible release before bridge parity tests."
        )
