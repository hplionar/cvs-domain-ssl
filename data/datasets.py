"""PyTorch datasets for Endoscapes SSL and CVS prediction."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Literal

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


DatasetMode = Literal["ssl", "supervised"]
VALID_SPLITS = {"train", "val", "test"}
VALID_MODES = {"ssl", "supervised"}
TARGET_COLUMNS = ["c1_consensus", "c2_consensus", "c3_consensus"]

REQUIRED_COLUMNS = {
    "sample_id",
    "video_id",
    "frame_num",
    "sequence_index",
    "split",
    "relative_path",
    "is_cvs_annotated",
    *TARGET_COLUMNS,
}


class EndoscapesDataset(Dataset):
    """Load Endoscapes frames from the validated project manifest.

    Modes:
        ssl:
            Uses every frame in the selected split and returns no CVS target.
        supervised:
            Uses only manually annotated keyframes and returns the three
            majority-consensus CVS criterion targets.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        dataset_root: str | Path,
        split: str,
        mode: DatasetMode,
        transform: Callable[[Image.Image], Any] | None = None,
        verify_files: bool = False,
    ) -> None:
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self.split = split
        self.mode = mode
        self.transform = transform

        self._validate_arguments()
        self.samples = self._load_samples()

        if verify_files:
            self._verify_files_exist()

    def _validate_arguments(self) -> None:
        if self.split not in VALID_SPLITS:
            raise ValueError(
                f"Invalid split {self.split!r}. Expected one of "
                f"{sorted(VALID_SPLITS)}."
            )
        if self.mode not in VALID_MODES:
            raise ValueError(
                f"Invalid mode {self.mode!r}. Expected one of "
                f"{sorted(VALID_MODES)}."
            )
        if not self.manifest_path.is_file():
            raise FileNotFoundError(
                f"Manifest file not found: {self.manifest_path}"
            )
        if not self.dataset_root.is_dir():
            raise FileNotFoundError(
                f"Endoscapes root not found: {self.dataset_root}"
            )

    def _load_samples(self) -> pd.DataFrame:
        manifest = pd.read_csv(self.manifest_path)
        missing_columns = REQUIRED_COLUMNS - set(manifest.columns)
        if missing_columns:
            raise ValueError(
                f"Manifest is missing columns: {sorted(missing_columns)}"
            )

        samples = manifest.loc[manifest["split"] == self.split].copy()

        if self.mode == "supervised":
            samples = samples.loc[samples["is_cvs_annotated"]].copy()
            if samples[TARGET_COLUMNS].isna().any().any():
                raise ValueError(
                    "A supervised sample is missing a consensus target."
                )

        if samples.empty:
            raise ValueError(
                f"No samples found for split={self.split!r}, mode={self.mode!r}."
            )

        samples = samples.sort_values(
            ["video_id", "sequence_index"], kind="stable"
        ).reset_index(drop=True)

        if samples["sample_id"].duplicated().any():
            raise ValueError("Filtered manifest contains duplicate sample IDs.")

        return samples

    def _verify_files_exist(self) -> None:
        missing = [
            str(self.dataset_root / relative_path)
            for relative_path in self.samples["relative_path"]
            if not (self.dataset_root / relative_path).is_file()
        ]
        if missing:
            preview = "\n".join(missing[:10])
            raise FileNotFoundError(
                f"{len(missing)} dataset images are missing. First entries:\n"
                f"{preview}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.samples.iloc[index]
        image_path = self.dataset_root / row["relative_path"]

        try:
            with Image.open(image_path) as image_file:
                image = image_file.convert("RGB")
        except FileNotFoundError as error:
            raise FileNotFoundError(
                f"Image for sample {row['sample_id']} not found: {image_path}"
            ) from error

        if self.transform is not None:
            image = self.transform(image)

        sample: dict[str, Any] = {
            "image": image,
            "sample_id": row["sample_id"],
            "video_id": int(row["video_id"]),
            "frame_num": int(row["frame_num"]),
            "sequence_index": int(row["sequence_index"]),
            "split": row["split"],
            "relative_path": row["relative_path"],
        }

        if self.mode == "supervised":
            sample["target"] = torch.tensor(
                row[TARGET_COLUMNS].to_numpy(dtype="float32"),
                dtype=torch.float32,
            )

        return sample

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"split={self.split!r}, "
            f"mode={self.mode!r}, "
            f"samples={len(self)})"
        )
