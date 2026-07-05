from __future__ import annotations

from pathlib import Path

from torch.utils.data import Dataset

from geowire.data.manifest import ClipRecord, load_manifest


class SpatialSFTDataset(Dataset):
    def __init__(self, items: list[ClipRecord]) -> None:
        self.items = items

    @classmethod
    def from_manifest(cls, path: str | Path) -> "SpatialSFTDataset":
        records = load_manifest(path)
        missing = [record.clip_id for record in records if not record.question or not record.answer]
        if missing:
            raise ValueError(f"Phase 2 QA records require question and answer; missing in {missing[:5]}")
        return cls(records)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> ClipRecord:
        return self.items[index]
