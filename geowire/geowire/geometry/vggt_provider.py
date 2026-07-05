from __future__ import annotations

from pathlib import Path


class VGGTProvider:
    """Thin placeholder for pinned VGGT integration.

    Real VGGT execution is intentionally isolated here so the main GeoWire code does
    not edit upstream VGGT files or depend on VGGT hidden states as semantic values.
    """

    def __init__(self, checkpoint: str | Path) -> None:
        self.checkpoint = str(checkpoint)

    def assert_available(self) -> None:
        path = Path(self.checkpoint)
        if path.exists():
            return
        if "/" in self.checkpoint:
            return
        raise FileNotFoundError(f"VGGT checkpoint is not available: {self.checkpoint}")

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "Real VGGT cache generation is gated behind M0/M1 inspection. "
            "Use scripts/cache_vggt.py after installing the pinned VGGT source."
        )
