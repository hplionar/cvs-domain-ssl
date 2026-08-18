"""Unlabelled frames for image-level continued pretraining.

The image counterpart to ``SSLClipDataset``, and deliberately close to it in
shape: a directory per procedure, a fixed quota drawn evenly across each, an
exclusion list wired to ``data.leakage``, and a deterministic index built once.

Why not ``SAGESFrameDataset``
-----------------------------
That class reads the labelled manifest and requires fifteen columns including
CVS consensus values. The SSL corpus in ``derived_ssl/`` has no labels at all --
it is 63,000 or 315,000 frames extracted at 1 or 5 fps by the challenge's own
``preprocess_videos.py`` -- so a manifest would have to be fabricated with empty
label columns purely to satisfy a reader that then discards them.

Why a quota at all
------------------
Every SAGES procedure is exactly 90 seconds, so at a fixed extraction rate each
yields the same number of frames and the quota does nothing. It exists for what
comes later: Cholec80 procedures average 29 minutes with a range of roughly 15
to 100, so a flat enumeration would let a handful of long procedures dominate a
combined corpus. Sampling at procedure level with a fixed quota keeps pools
equal in size by construction.

``frames_per_video=None`` uses every frame, which is the right default while the
corpus is SAGES alone.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable, NamedTuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class FrameIndex(NamedTuple):
    """One frame, resolved before training starts."""

    video_id: str
    path: Path


class SSLFrameDataset(Dataset):
    """Unlabelled frames drawn from a directory of per-procedure folders.

    Expects the layout ``preprocess_videos.py`` produces::

        frames_dir/
            <video_id>/
                frame_0000.jpg
                frame_0001.jpg
                ...

    Parameters
    ----------
    frames_dir:
        Directory containing one subdirectory per procedure.
    frames_per_video:
        Fixed quota per procedure, spread evenly across it. ``None`` uses every
        frame.
    exclude_video_ids:
        Identifiers to drop, normally supplied by ``data.leakage``. Kept even
        though SSL uses no labels: a procedure appearing in an evaluation set
        should not be pretrained on, or a later reader of the results cannot
        tell whether the representation had seen its test data.
    limit_videos:
        Truncate the corpus. This is how the overfitting check is built -- a
        handful of procedures seen hundreds of times. If the loss does not
        collapse there, the problem is not the learning rate.
    """

    def __init__(
        self,
        frames_dir: str | Path,
        *,
        frames_per_video: int | None = None,
        transform: Callable | None = None,
        extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png"),
        exclude_video_ids: Iterable[str] | None = None,
        limit_videos: int | None = None,
    ) -> None:
        self.frames_dir = Path(frames_dir)
        self.frames_per_video = frames_per_video
        self.transform = transform
        self.extensions = tuple(e.lower() for e in extensions)

        if not self.frames_dir.is_dir():
            raise FileNotFoundError(f"No such directory: {self.frames_dir}")

        excluded = {str(v) for v in (exclude_video_ids or ())}
        video_dirs = sorted(
            d for d in self.frames_dir.iterdir()
            if d.is_dir() and d.name not in excluded
        )
        if not video_dirs:
            raise ValueError(
                f"No procedure directories in {self.frames_dir}. Expected one "
                f"subdirectory per video, each containing extracted frames."
            )

        if limit_videos is not None:
            video_dirs = video_dirs[:limit_videos]

        self.limit_videos = limit_videos
        self.excluded_video_ids = sorted(excluded)
        self.video_ids = [d.name for d in video_dirs]

        self.frames: list[FrameIndex] = []
        self.short_videos: list[tuple[str, int]] = []
        for directory in video_dirs:
            paths = sorted(
                p for p in directory.iterdir()
                if p.suffix.lower() in self.extensions
            )
            if not paths:
                continue

            if frames_per_video is None or frames_per_video >= len(paths):
                if frames_per_video is not None and frames_per_video > len(paths):
                    # Reported rather than silently padded or repeated: a quota
                    # the corpus cannot meet makes pools unequal, which is the
                    # thing the quota exists to prevent.
                    self.short_videos.append((directory.name, len(paths)))
                selected = paths
            else:
                # Even spread, not random: a quota should cover the whole
                # procedure rather than clustering in one phase of it.
                idx = np.linspace(0, len(paths) - 1, frames_per_video)
                selected = [paths[int(round(i))] for i in idx]

            self.frames.extend(FrameIndex(directory.name, p) for p in selected)

        if not self.frames:
            raise ValueError(
                f"Found {len(video_dirs)} procedure directories in "
                f"{self.frames_dir} but no frames with extensions "
                f"{self.extensions}."
            )

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, index: int) -> Any:
        entry = self.frames[index]
        with Image.open(entry.path) as handle:
            image = handle.convert("RGB")
        if self.transform is not None:
            return self.transform(image)
        return image

    def describe(self) -> dict[str, Any]:
        """Recorded alongside a pretraining run so the corpus is reconstructable."""
        record: dict[str, Any] = {
            "frames_dir": str(self.frames_dir),
            "num_videos": len(self.video_ids),
            "num_frames": len(self.frames),
            "frames_per_video": self.frames_per_video,
            "limit_videos": self.limit_videos,
            "excluded_video_ids": self.excluded_video_ids,
        }
        if self.short_videos:
            record["videos_below_quota"] = self.short_videos[:20]
            record["num_videos_below_quota"] = len(self.short_videos)
        return record

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(videos={len(self.video_ids)}, "
            f"frames={len(self.frames)}, quota={self.frames_per_video})"
        )


__all__ = ["FrameIndex", "SSLFrameDataset"]
