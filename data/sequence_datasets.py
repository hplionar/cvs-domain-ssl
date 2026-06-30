from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, Literal

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class EndoscapesClipDataset(Dataset):
    """
    Endoscapes clip dataset for temporal CVS experiments.

    In supervised mode:
        - each sample is centred on, or ends at, a manually annotated CVS keyframe
        - the target is the C1/C2/C3 consensus label of that annotated keyframe only
        - surrounding frames provide visual context but are not treated as labels

    In ssl mode:
        - clips can be sampled from all frames in the selected split
        - no CVS target is returned

    Returned frame tensor shape after transforms:
        [T, C, H, W]
    """

    VALID_SPLITS = {"train", "val", "test"}
    VALID_MODES = {"supervised", "ssl"}
    VALID_TARGET_POSITIONS = {"end", "center"}

    TARGET_COLUMNS = ["c1_consensus", "c2_consensus", "c3_consensus"]

    def __init__(
        self,
        manifest_path: str | Path,
        dataset_root: str | Path,
        split: Literal["train", "val", "test"],
        mode: Literal["supervised", "ssl"] = "supervised",
        clip_length: int = 5,
        stride: int = 1,
        target_position: Literal["end", "center"] = "end",
        transform: Any | None = None,
        verify_images: bool = False,
        consistent_transform: bool = True,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.mode = mode
        self.clip_length = clip_length
        self.stride = stride
        self.target_position = target_position
        self.transform = transform
        self.verify_images = verify_images
        self.consistent_transform = consistent_transform

        if split not in self.VALID_SPLITS:
            raise ValueError(f"Invalid split: {split}. Expected one of {self.VALID_SPLITS}")

        if mode not in self.VALID_MODES:
            raise ValueError(f"Invalid mode: {mode}. Expected one of {self.VALID_MODES}")

        if target_position not in self.VALID_TARGET_POSITIONS:
            raise ValueError(
                f"Invalid target_position: {target_position}. "
                f"Expected one of {self.VALID_TARGET_POSITIONS}"
            )

        if clip_length < 1:
            raise ValueError("clip_length must be >= 1")

        if stride < 1:
            raise ValueError("stride must be >= 1")

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")

        if not self.dataset_root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.dataset_root}")

        self.manifest = pd.read_csv(self.manifest_path)
        self._validate_manifest_columns()

        self.frames = self.manifest[self.manifest["split"] == split].copy()

        if self.frames.empty:
            raise ValueError(f"No frames found for split={split}")

        self.frames["video_id"] = self.frames["video_id"].astype(int)
        self.frames["frame_num"] = self.frames["frame_num"].astype(int)
        self.frames["sequence_index"] = self.frames["sequence_index"].astype(int)

        self.frames = self.frames.sort_values(
            ["video_id", "sequence_index"]
        ).reset_index(drop=True)

        if mode == "supervised":
            self.targets = self.frames[self.frames["is_cvs_annotated"] == True].copy()

            if self.targets.empty:
                raise ValueError(f"No annotated CVS keyframes found for split={split}")

            if self.targets[self.TARGET_COLUMNS].isna().any().any():
                raise ValueError("Supervised target rows contain missing consensus labels.")
        else:
            self.targets = self.frames.copy()

        self.targets = self.targets.reset_index(drop=True)

        self.video_groups: dict[int, pd.DataFrame] = {
            int(video_id): group.sort_values("sequence_index").reset_index(drop=True)
            for video_id, group in self.frames.groupby("video_id")
        }

        self.sequence_lookup: dict[int, dict[int, int]] = {}
        for video_id, group in self.video_groups.items():
            self.sequence_lookup[video_id] = {
                int(row.sequence_index): int(i)
                for i, row in group.iterrows()
            }

        if verify_images:
            self._verify_referenced_images()

    def _validate_manifest_columns(self) -> None:
        required = {
            "sample_id",
            "video_id",
            "frame_num",
            "sequence_index",
            "split",
            "relative_path",
            "is_cvs_annotated",
            *self.TARGET_COLUMNS,
        }

        missing = required.difference(self.manifest.columns)

        if missing:
            raise ValueError(f"Manifest missing required columns: {sorted(missing)}")

    def _verify_referenced_images(self) -> None:
        missing = []

        for relative_path in self.frames["relative_path"]:
            path = self.dataset_root / relative_path
            if not path.exists():
                missing.append(str(path))

        if missing:
            preview = "\n".join(missing[:10])
            raise FileNotFoundError(
                f"Found {len(missing)} missing image files. First examples:\n{preview}"
            )

    def __len__(self) -> int:
        return len(self.targets)

    def _clip_sequence_indices(self, target_sequence_index: int) -> list[int]:
        if self.target_position == "end":
            offsets = list(range(-(self.clip_length - 1), 1))
        elif self.target_position == "center":
            left = self.clip_length // 2
            right = self.clip_length - left - 1
            offsets = list(range(-left, right + 1))
        else:
            raise ValueError(f"Unsupported target_position: {self.target_position}")

        return [
            target_sequence_index + offset * self.stride
            for offset in offsets
        ]

    def _get_frame_row(
        self,
        video_id: int,
        sequence_index: int,
    ) -> pd.Series:
        group = self.video_groups[video_id]
        lookup = self.sequence_lookup[video_id]

        min_index = int(group["sequence_index"].min())
        max_index = int(group["sequence_index"].max())

        # Boundary handling: replicate nearest available frame.
        sequence_index = max(min_index, min(max_index, sequence_index))

        if sequence_index not in lookup:
            # This should not normally happen because sequence_index is continuous
            # in our generated manifest, but nearest fallback keeps the dataset robust.
            available = np.array(sorted(lookup.keys()))
            nearest = int(available[np.argmin(np.abs(available - sequence_index))])
            sequence_index = nearest

        row_position = lookup[sequence_index]
        return group.iloc[row_position]

    def _load_image(self, relative_path: str) -> Image.Image:
        path = self.dataset_root / relative_path

        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        return Image.open(path).convert("RGB")

    def _apply_transform_to_clip(self, images: list[Image.Image]) -> torch.Tensor | list[Any]:
        if self.transform is None:
            return images

        transformed = []

        if self.consistent_transform:
            seed = random.randint(0, 2**32 - 1)

            for image in images:
                random.seed(seed)
                np.random.seed(seed % (2**32 - 1))
                torch.manual_seed(seed)
                transformed.append(self.transform(image))
        else:
            transformed = [self.transform(image) for image in images]

        if all(isinstance(frame, torch.Tensor) for frame in transformed):
            return torch.stack(transformed, dim=0)

        return transformed

    def __getitem__(self, index: int) -> Dict[str, Any]:
        target_row = self.targets.iloc[index]

        video_id = int(target_row["video_id"])
        target_sequence_index = int(target_row["sequence_index"])

        sequence_indices = self._clip_sequence_indices(target_sequence_index)

        clip_rows = [
            self._get_frame_row(video_id, sequence_index)
            for sequence_index in sequence_indices
        ]

        images = [
            self._load_image(str(row["relative_path"]))
            for row in clip_rows
        ]

        frames = self._apply_transform_to_clip(images)

        sample: Dict[str, Any] = {
            "frames": frames,
            "target_sample_id": str(target_row["sample_id"]),
            "sample_ids": [str(row["sample_id"]) for row in clip_rows],
            "video_id": video_id,
            "frame_nums": torch.tensor(
                [int(row["frame_num"]) for row in clip_rows],
                dtype=torch.long,
            ),
            "sequence_indices": torch.tensor(
                [int(row["sequence_index"]) for row in clip_rows],
                dtype=torch.long,
            ),
            "split": self.split,
        }

        if self.mode == "supervised":
            target = target_row[self.TARGET_COLUMNS].to_numpy(dtype=np.float32)
            sample["target"] = torch.tensor(target, dtype=torch.float32)

        return sample

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"split={self.split!r}, "
            f"mode={self.mode!r}, "
            f"samples={len(self)}, "
            f"clip_length={self.clip_length}, "
            f"stride={self.stride}, "
            f"target_position={self.target_position!r})"
        )
