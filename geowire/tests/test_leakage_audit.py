from __future__ import annotations

from geowire.data.leakage_audit import audit_scene_leakage
from geowire.types import ClipRecord


def rec(clip_id: str, scene_id: str, split: str) -> ClipRecord:
    return ClipRecord(
        clip_id=clip_id,
        scene_id=scene_id,
        source_dataset="toy",
        frame_paths=("a.png",),
        frame_indices=(0,),
        timestamps_s=(0.0,),
        split=split,  # type: ignore[arg-type]
    )


def test_leakage_detects_overlap() -> None:
    report = audit_scene_leakage([rec("a", "s1", "train"), rec("b", "s1", "test")])
    assert not report.passed
    assert report.overlapping_scenes == ("s1",)


def test_leakage_passes_disjoint() -> None:
    report = audit_scene_leakage([rec("a", "s1", "train"), rec("b", "s2", "test")])
    assert report.passed
