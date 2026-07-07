from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class SAGESFrameDataset(Dataset):
    """Frame-level SAGES CVS dataset using the internal video-level split manifest."""

    VALID_SPLITS = {"train", "val", "test"}
    VALID_MODES = {"ssl", "supervised"}

    REQUIRED_COLUMNS = {
        "sample_id",
        "dataset",
        "internal_split",
        "video_id",
        "frame_id",
        "source_frame_id",
        "extracted_frame_id",
        "sequence_index",
        "timestamp_sec",
        "relative_path",
        "video_relative_path",
        "c1_consensus",
        "c2_consensus",
        "c3_consensus",
        "cvs_consensus",
    }

    def __init__(
        self,
        manifest_path: str | Path,
        dataset_root: str | Path,
        split: str = "train",
        mode: str = "supervised",
        transform: Any | None = None,
        verify_images: bool = False,
    ) -> None:
        if split not in self.VALID_SPLITS:
            raise ValueError(f"Invalid split '{split}'. Expected one of {sorted(self.VALID_SPLITS)}.")

        if mode not in self.VALID_MODES:
            raise ValueError(f"Invalid mode '{mode}'. Expected one of {sorted(self.VALID_MODES)}.")

        self.manifest_path = Path(manifest_path)
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.mode = mode
        self.transform = transform

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        if not self.dataset_root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.dataset_root}")

        df = pd.read_csv(self.manifest_path)

        missing = sorted(self.REQUIRED_COLUMNS.difference(df.columns))
        if missing:
            raise ValueError(f"Manifest missing required columns: {missing}")

        df = df[df["internal_split"] == split].copy()

        if df.empty:
            raise ValueError(f"No SAGES samples found for split='{split}'.")

        if df["sample_id"].duplicated().any():
            dupes = df[df["sample_id"].duplicated()]["sample_id"].head().tolist()
            raise ValueError(f"Duplicate sample_id values found: {dupes}")

        if mode == "supervised":
            target_cols = ["c1_consensus", "c2_consensus", "c3_consensus"]
            if df[target_cols].isna().any().any():
                raise ValueError("Supervised SAGES rows contain missing consensus targets.")

        df = df.sort_values(["video_id", "sequence_index"]).reset_index(drop=True)

        if verify_images:
            missing_paths = []
            for rel_path in df["relative_path"]:
                path = self.dataset_root / rel_path
                if not path.exists():
                    missing_paths.append(str(path))
                    if len(missing_paths) >= 10:
                        break

            if missing_paths:
                raise FileNotFoundError(f"Missing image files, examples: {missing_paths}")

        self.df = df

    def __len__(self) -> int:
        return len(self.df)

    def __repr__(self) -> str:
        return (
            f"SAGESFrameDataset(split='{self.split}', "
            f"mode='{self.mode}', samples={len(self)})"
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.df.iloc[index]

        image_path = self.dataset_root / row["relative_path"]
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        sample: dict[str, Any] = {
            "image": image,
            "sample_id": str(row["sample_id"]),
            "dataset": "sages",
            "split": str(row["internal_split"]),
            "official_split": str(row["official_split"]),
            "video_id": str(row["video_id"]),
            "frame_id": int(row["frame_id"]),
            "source_frame_id": int(row["source_frame_id"]),
            "extracted_frame_id": int(row["extracted_frame_id"]),
            "sequence_index": int(row["sequence_index"]),
            "timestamp_sec": float(row["timestamp_sec"]),
            "relative_path": str(row["relative_path"]),
            "video_relative_path": str(row["video_relative_path"]),
        }

        if self.mode == "supervised":
            target = torch.tensor(
                [
                    float(row["c1_consensus"]),
                    float(row["c2_consensus"]),
                    float(row["c3_consensus"]),
                ],
                dtype=torch.float32,
            )
            sample["target"] = target

        return sample
