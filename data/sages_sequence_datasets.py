from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class SAGESClipDataset(Dataset):
    """Sparse temporal SAGES clip dataset from labelled 5-second derived frames.

    This dataset uses the validated SAGES internal-split manifest and constructs
    clips from derived frames:
        frame_0000, frame_0001, ..., frame_0017

    These clips are sparse 5-second-step clips, not dense 30-fps video clips.
    Dense MP4 clip decoding can be added later as a separate dataset.
    """

    VALID_SPLITS = {"train", "val", "test"}
    VALID_MODES = {"ssl", "supervised"}
    VALID_TARGET_POSITIONS = {"end", "center"}

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
        clip_length: int = 5,
        stride: int = 1,
        target_position: str = "end",
        transform: Any | None = None,
        consistent_transform: bool = True,
        verify_images: bool = False,
    ) -> None:
        if split not in self.VALID_SPLITS:
            raise ValueError(f"Invalid split '{split}'. Expected one of {sorted(self.VALID_SPLITS)}.")

        if mode not in self.VALID_MODES:
            raise ValueError(f"Invalid mode '{mode}'. Expected one of {sorted(self.VALID_MODES)}.")

        if target_position not in self.VALID_TARGET_POSITIONS:
            raise ValueError(
                f"Invalid target_position '{target_position}'. "
                f"Expected one of {sorted(self.VALID_TARGET_POSITIONS)}."
            )

        if clip_length <= 0:
            raise ValueError("clip_length must be positive.")

        if stride <= 0:
            raise ValueError("stride must be positive.")

        self.manifest_path = Path(manifest_path)
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.mode = mode
        self.clip_length = clip_length
        self.stride = stride
        self.target_position = target_position
        self.transform = transform
        self.consistent_transform = consistent_transform

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

        self.video_groups: dict[str, pd.DataFrame] = {
            str(video_id): group.sort_values("sequence_index").reset_index(drop=True)
            for video_id, group in self.df.groupby("video_id")
        }

    def __len__(self) -> int:
        return len(self.df)

    def __repr__(self) -> str:
        return (
            f"SAGESClipDataset(split='{self.split}', mode='{self.mode}', "
            f"samples={len(self)}, clip_length={self.clip_length}, "
            f"stride={self.stride}, target_position='{self.target_position}')"
        )

    def _clip_offsets(self) -> list[int]:
        if self.target_position == "end":
            start = -(self.clip_length - 1) * self.stride
            return list(range(start, 1, self.stride))

        center = self.clip_length // 2
        return [
            (i - center) * self.stride
            for i in range(self.clip_length)
        ]

    def _load_image(self, relative_path: str) -> Image.Image:
        image_path = self.dataset_root / relative_path
        return Image.open(image_path).convert("RGB")

    def _apply_transform(self, images: list[Image.Image]) -> list[Any]:
        if self.transform is None:
            return images

        if not self.consistent_transform:
            return [self.transform(image) for image in images]

        seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
        transformed = []

        for image in images:
            torch.manual_seed(seed)
            transformed.append(self.transform(image))

        return transformed

    def __getitem__(self, index: int) -> dict[str, Any]:
        target_row = self.df.iloc[index]

        video_id = str(target_row["video_id"])
        video_df = self.video_groups[video_id]

        target_sequence_index = int(target_row["sequence_index"])
        max_index = len(video_df) - 1

        offsets = self._clip_offsets()
        clip_sequence_indices = [
            min(max(target_sequence_index + offset, 0), max_index)
            for offset in offsets
        ]

        clip_rows = video_df.iloc[clip_sequence_indices]

        images = [
            self._load_image(str(row["relative_path"]))
            for _, row in clip_rows.iterrows()
        ]

        frames = self._apply_transform(images)

        if isinstance(frames[0], torch.Tensor):
            frames_out = torch.stack(frames, dim=0)
        else:
            frames_out = frames

        sample: dict[str, Any] = {
            "frames": frames_out,
            "sample_ids": [str(x) for x in clip_rows["sample_id"].tolist()],
            "target_sample_id": str(target_row["sample_id"]),
            "dataset": "sages",
            "split": str(target_row["internal_split"]),
            "official_split": str(target_row["official_split"]),
            "video_id": video_id,
            "frame_ids": torch.tensor(clip_rows["frame_id"].to_numpy(), dtype=torch.long),
            "source_frame_ids": torch.tensor(clip_rows["source_frame_id"].to_numpy(), dtype=torch.long),
            "extracted_frame_ids": torch.tensor(clip_rows["extracted_frame_id"].to_numpy(), dtype=torch.long),
            "sequence_indices": torch.tensor(clip_rows["sequence_index"].to_numpy(), dtype=torch.long),
            "timestamps_sec": torch.tensor(clip_rows["timestamp_sec"].to_numpy(), dtype=torch.float32),
            "target_frame_id": int(target_row["frame_id"]),
            "target_sequence_index": target_sequence_index,
            "target_timestamp_sec": float(target_row["timestamp_sec"]),
            "video_relative_path": str(target_row["video_relative_path"]),
        }

        if self.mode == "supervised":
            target = torch.tensor(
                [
                    float(target_row["c1_consensus"]),
                    float(target_row["c2_consensus"]),
                    float(target_row["c3_consensus"]),
                ],
                dtype=torch.float32,
            )
            sample["target"] = target

        return sample
