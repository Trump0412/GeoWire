from __future__ import annotations

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from geowire.types import ClipRecord


class ClipDataset(Dataset):
    def __init__(self, records: list[ClipRecord]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        images = [Image.open(Path(path)).convert("RGB") for path in record.frame_paths]
        return {"record": record, "images": images}
