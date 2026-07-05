from __future__ import annotations


def validate_lora_targets(module_names: list[str], target_patterns: list[str]) -> list[str]:
    matched: list[str] = []
    for pattern in target_patterns:
        needle = pattern.replace("*", "")
        hits = [name for name in module_names if needle in name]
        if not hits:
            raise ValueError(f"LoRA target pattern matched no modules: {pattern}")
        matched.extend(hits)
    return sorted(set(matched))
