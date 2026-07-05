from __future__ import annotations

from dataclasses import dataclass

from geowire.types import ClipRecord


@dataclass(frozen=True)
class LeakageReport:
    train_scenes: int
    test_scenes: int
    overlapping_scenes: tuple[str, ...]
    missing_scene_records: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.overlapping_scenes and not self.missing_scene_records


def audit_scene_leakage(records: list[ClipRecord]) -> LeakageReport:
    missing = tuple(sorted(r.clip_id for r in records if not r.scene_id))
    train = {r.scene_id for r in records if r.split == "train" and r.scene_id}
    test = {r.scene_id for r in records if r.split in {"val", "test"} and r.scene_id}
    overlap = tuple(sorted(train & test))
    return LeakageReport(
        train_scenes=len(train),
        test_scenes=len(test),
        overlapping_scenes=overlap,
        missing_scene_records=missing,
    )
