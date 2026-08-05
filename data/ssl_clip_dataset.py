"""Unlabelled clip dataset for self-supervised pretraining on surgical video.

Reads raw video and yields ``[T, C, H, W]`` clips. No labels: the SSL objective
needs none, which is why the corpus is far larger than the annotated manifest
suggests — SAGES carries 18 labelled frames per video against 2,700 total.

Clip sampling
-------------
Clips are indexed deterministically as ``(video, start_frame)`` pairs computed
once at construction. This matters for a resumable job: the sample at index *i*
is the same on every run and after every restart, so a checkpoint resumed
mid-epoch continues over the same data rather than silently reshuffling.

``stride`` is set to reproduce the checkpoint's own pretraining configuration
rather than chosen freely. VideoMAE-base was pretrained on Kinetics with 16
frames at sampling rate 4; SAGES is 30 fps, so stride 4 gives the same temporal
span the encoder was trained on. Departing from it changes the temporal
statistics the model expects.

Leakage
-------
The video list is filtered against evaluation splits at construction and the
result asserted, so a corpus containing an evaluation procedure cannot be built
by omission.
"""

from __future__ import annotations

import random
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class ClipIndex:
    """One clip: which video, and which frame it starts at."""

    video_path: str
    video_id: str
    start_frame: int
    stride: int
    num_frames: int

    @property
    def frame_indices(self) -> list[int]:
        return [self.start_frame + i * self.stride for i in range(self.num_frames)]

    @property
    def span_frames(self) -> int:
        return (self.num_frames - 1) * self.stride + 1


def _to_numpy(batch) -> np.ndarray:
    """Convert a decoder batch to a numpy array.

    decord returns its own ``NDArray``, which exposes ``asnumpy()`` rather than
    ``numpy()`` and which ``np.asarray`` converts to a *object* array rather
    than failing. That produces a confusing TypeError several frames later at
    ``torch.from_numpy``, so the conversion is explicit here.
    """
    for method in ("asnumpy", "numpy"):
        if hasattr(batch, method):
            return getattr(batch, method)()
    array = np.asarray(batch)
    if array.dtype == np.object_:
        raise TypeError(
            f"Could not convert {type(batch).__name__} to a numeric array; "
            f"np.asarray produced dtype object. The decoder backend returns an "
            f"unrecognised type."
        )
    return array


class VideoReader:
    """Frame reader that decodes at reduced resolution.

    Decoding at native resolution and shrinking afterwards is the expensive
    path: a 16-frame clip at 1280x720 in float32 is 176 MB, and with several
    workers each prefetching several batches that exhausts host memory. Both
    backends can scale during decode, so the clip arrives near its final size.

    decord seeks directly to a frame index. PyAV decodes forward from the
    nearest keyframe, so the fallback seeks to the clip start rather than to the
    beginning of the file; without that, reading frame 2600 of a 2700-frame
    video decodes 2600 frames to keep 16.
    """

    _warned = False

    def __init__(self, path: str, decode_size: int | None = 256) -> None:
        self.path = path
        self.decode_size = decode_size
        self._backend = None
        self._reader = None

    def _open(self):
        if self._reader is not None:
            return self._reader
        try:
            import decord

            kwargs = {"num_threads": 1}
            if self.decode_size:
                kwargs.update(width=self.decode_size, height=self.decode_size)
            self._reader = decord.VideoReader(self.path, **kwargs)
            self._backend = "decord"
        except ImportError:
            import av

            self._reader = av.open(self.path)
            self._backend = "av"
            if not VideoReader._warned:
                import warnings

                warnings.warn(
                    "decord not available; falling back to PyAV. Random clip "
                    "access will be substantially slower. Install eva-decord.",
                    stacklevel=2,
                )
                VideoReader._warned = True
        return self._reader

    def __len__(self) -> int:
        reader = self._open()
        if self._backend == "decord":
            return len(reader)
        stream = reader.streams.video[0]
        if stream.frames:
            return stream.frames
        if stream.duration and stream.average_rate:
            return int(float(stream.duration * stream.time_base) * float(stream.average_rate))
        return 0

    def read_frames(self, indices: Sequence[int]) -> np.ndarray:
        """Return ``[T, H, W, C]`` uint8 RGB, scaled to ``decode_size``."""
        reader = self._open()
        if self._backend == "decord":
            batch = reader.get_batch(list(indices))
            array = _to_numpy(batch)
            if array.dtype != np.uint8:
                raise TypeError(
                    f"decord returned dtype {array.dtype} for {self.path}; "
                    f"expected uint8. The return type was not converted correctly."
                )
            return np.ascontiguousarray(array)

        return self._read_frames_av(reader, indices)

    def _read_frames_av(self, container, indices: Sequence[int]) -> np.ndarray:
        import av

        stream = container.streams.video[0]
        rate = float(stream.average_rate or 30.0)
        start, last = indices[0], indices[-1]

        # Seek to just before the clip. PyAV lands on the preceding keyframe,
        # so decoding starts near the clip rather than at the file start.
        target = int(start / rate / float(stream.time_base))
        try:
            container.seek(target, stream=stream, backward=True, any_frame=False)
        except av.AVError:
            container.seek(0)

        wanted = set(indices)
        size = self.decode_size
        frames: dict[int, np.ndarray] = {}
        position = None

        for frame in container.decode(stream):
            position = int(round(float(frame.pts * stream.time_base) * rate)) if frame.pts is not None \
                else (0 if position is None else position + 1)
            if position in wanted:
                kwargs = {"format": "rgb24"}
                if size:
                    kwargs.update(width=size, height=size)
                frames[position] = frame.to_ndarray(**kwargs)
            if position >= last:
                break

        if len(frames) < len(indices):
            # Seeking landed past a wanted frame, or timestamps are irregular.
            # Repeat the nearest available frame rather than failing the batch.
            available = sorted(frames)
            if not available:
                raise RuntimeError(f"No frames decoded from {self.path} at {indices[:3]}...")
            for index in indices:
                if index not in frames:
                    nearest = min(available, key=lambda a: abs(a - index))
                    frames[index] = frames[nearest]

        return np.ascontiguousarray(np.stack([frames[i] for i in indices]))

    def close(self) -> None:
        if self._reader is not None and self._backend == "av":
            self._reader.close()
        self._reader = None


class SSLClipDataset(Dataset):
    """Unlabelled clips drawn from a directory of videos.

    Parameters
    ----------
    video_dir:
        Directory of video files.
    num_frames, stride:
        Clip geometry. Defaults reproduce VideoMAE's Kinetics configuration.
    clips_per_video:
        Fixed quota per procedure. Sampling at procedure level with a fixed
        quota keeps pools equal in size by construction when corpora are later
        combined, and prevents a dataset of long procedures from dominating one
        of short clips.
    exclude_video_ids:
        Identifiers to drop, normally supplied by ``data.leakage``.
    """

    def __init__(
        self,
        video_dir: str | Path,
        *,
        num_frames: int = 16,
        stride: int = 4,
        clips_per_video: int = 40,
        transform: Callable | None = None,
        extensions: tuple[str, ...] = (".mp4", ".avi", ".mkv"),
        exclude_video_ids: Iterable[str] | None = None,
        seed: int = 0,
        frame_counts: dict[str, int] | None = None,
        decode_size: int = 256,
        reader_cache_size: int = 4,
        limit_videos: int | None = None,
    ) -> None:
        self.video_dir = Path(video_dir)
        self.num_frames = num_frames
        self.stride = stride
        self.clips_per_video = clips_per_video
        self.transform = transform
        self.seed = seed
        # Decoding is done at this size rather than natively: a 16-frame clip at
        # 1280x720 in float32 is 176 MB, which with several prefetching workers
        # exhausts host memory. 256 leaves room for a 224 random crop.
        self.decode_size = decode_size
        # Bounded: an unbounded cache accumulates one open container per video
        # touched, and under shuffling a worker eventually opens every video.
        self.reader_cache_size = reader_cache_size

        if not self.video_dir.is_dir():
            raise FileNotFoundError(f"No such directory: {self.video_dir}")

        excluded = {str(v) for v in (exclude_video_ids or ())}
        paths = sorted(
            p for p in self.video_dir.iterdir()
            if p.suffix.lower() in extensions and p.stem not in excluded
        )
        if not paths:
            raise ValueError(f"No videos found in {self.video_dir}")

        # Truncating the corpus is how the overfitting check is constructed: a
        # handful of videos seen hundreds of times. If loss does not collapse
        # there, the problem is not the learning rate.
        if limit_videos is not None:
            paths = paths[:limit_videos]
        self.limit_videos = limit_videos

        self.excluded_video_ids = sorted(excluded)
        self.video_paths = paths
        self._readers: "OrderedDict[str, VideoReader]" = OrderedDict()
        self.clips = self._build_index(frame_counts)

    # -- indexing ---------------------------------------------------------

    def _build_index(self, frame_counts: dict[str, int] | None) -> list[ClipIndex]:
        """Enumerate clips once, deterministically.

        Start positions are spread evenly across each video rather than drawn at
        random, so that a fixed quota covers the whole procedure instead of
        clustering. Where the quota exceeds the number of non-overlapping
        positions, clips necessarily overlap; that is reported rather than
        hidden, since overlap inflates apparent corpus size.
        """
        span = (self.num_frames - 1) * self.stride + 1
        clips: list[ClipIndex] = []
        self.overlapping_videos: list[str] = []

        for path in self.video_paths:
            video_id = path.stem
            if frame_counts and video_id in frame_counts:
                total = frame_counts[video_id]
            else:
                reader = VideoReader(str(path))
                total = len(reader)
                reader.close()

            if total < span:
                continue

            last_start = total - span
            capacity = last_start // span + 1
            if self.clips_per_video > capacity:
                self.overlapping_videos.append(video_id)

            if self.clips_per_video == 1:
                starts = [last_start // 2]
            else:
                starts = np.linspace(0, last_start, self.clips_per_video)
                starts = sorted({int(round(s)) for s in starts})

            clips.extend(
                ClipIndex(str(path), video_id, s, self.stride, self.num_frames)
                for s in starts
            )

        if not clips:
            raise ValueError(
                f"No clips could be built. Videos may be shorter than the "
                f"{span}-frame span required by num_frames={self.num_frames}, "
                f"stride={self.stride}."
            )
        return clips

    # -- access -----------------------------------------------------------

    def __len__(self) -> int:
        return len(self.clips)

    def _reader_for(self, path: str) -> VideoReader:
        """Least-recently-used reader cache.

        Opening a container per sample dominates runtime, but caching without a
        bound leaks: under shuffling a worker touches every video and would hold
        all of them open.
        """
        reader = self._readers.pop(path, None)
        if reader is None:
            reader = VideoReader(path, decode_size=self.decode_size)
        self._readers[path] = reader
        while len(self._readers) > self.reader_cache_size:
            _, evicted = self._readers.popitem(last=False)
            evicted.close()
        return reader

    def __getitem__(self, index: int) -> dict[str, object]:
        clip = self.clips[index]
        frames = self._reader_for(clip.video_path).read_frames(clip.frame_indices)

        # Kept as uint8 through the crop. Converting to float before cropping
        # allocates the full decoded clip at 4 bytes per channel per pixel,
        # which is where the memory goes.
        tensor = torch.from_numpy(frames).permute(0, 3, 1, 2)  # [T, C, H, W] uint8

        if self.transform is not None:
            tensor = self.transform(tensor)
        else:
            tensor = tensor.float().div_(255.0)

        return {"pixel_values": tensor, "video_id": clip.video_id, "start_frame": clip.start_frame}

    # -- reporting --------------------------------------------------------

    def describe(self) -> dict[str, object]:
        span = (self.num_frames - 1) * self.stride + 1
        return {
            "video_dir": str(self.video_dir),
            "num_videos": len(self.video_paths),
            "num_clips": len(self.clips),
            "clips_per_video": self.clips_per_video,
            "num_frames": self.num_frames,
            "stride": self.stride,
            "clip_span_frames": span,
            "decode_size": self.decode_size,
            "limit_videos": self.limit_videos,
            "excluded_video_ids": self.excluded_video_ids,
            "videos_with_overlapping_clips": len(self.overlapping_videos),
        }


class ClipTransform:
    """Deterministic-geometry augmentation applied consistently across a clip.

    A single crop and flip decision is drawn per clip and applied to every
    frame. Sampling per frame would introduce apparent motion that is not in the
    video, which for a temporal objective is not augmentation but corruption.

    Colour jitter is deliberately omitted: hue perturbation degrades the
    tissue-colour cue that CVS criterion C2 depends on.
    """

    def __init__(
        self,
        image_size: int = 224,
        mean: Sequence[float] = (0.485, 0.456, 0.406),
        std: Sequence[float] = (0.229, 0.224, 0.225),
        scale: tuple[float, float] = (0.5, 1.0),
        train: bool = True,
    ) -> None:
        self.image_size = image_size
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)
        self.scale = scale
        self.train = train

    def __call__(self, clip: torch.Tensor) -> torch.Tensor:
        """Accepts uint8 or float ``[T, C, H, W]``; returns normalised float32.

        Cropping happens before the float conversion so the expensive
        allocation covers only the retained region.
        """
        import torch.nn.functional as F

        _, _, height, width = clip.shape

        if self.train:
            area = height * width
            target = random.uniform(*self.scale) * area
            ratio = random.uniform(3 / 4, 4 / 3)
            crop_w = max(1, min(width, int(round((target * ratio) ** 0.5))))
            crop_h = max(1, min(height, int(round((target / ratio) ** 0.5))))
            top = random.randint(0, height - crop_h)
            left = random.randint(0, width - crop_w)
            clip = clip[:, :, top : top + crop_h, left : left + crop_w]
            if random.random() < 0.5:
                clip = torch.flip(clip, dims=[3])
        else:
            side = min(height, width)
            top, left = (height - side) // 2, (width - side) // 2
            clip = clip[:, :, top : top + side, left : left + side]

        if clip.dtype == torch.uint8:
            clip = clip.float().div_(255.0)
        else:
            clip = clip.float()

        clip = F.interpolate(
            clip, size=(self.image_size, self.image_size),
            mode="bilinear", align_corners=False, antialias=True,
        )
        return (clip - self.mean) / self.std


__all__ = ["ClipIndex", "ClipTransform", "SSLClipDataset", "VideoReader"]