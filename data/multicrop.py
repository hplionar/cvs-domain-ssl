"""Multi-crop augmentation for DINO self-distillation.

DINO's local-to-global correspondence is the whole objective: the teacher sees
only wide views, the student sees those plus several narrow ones, and matching
them forces the student to infer scene-level content from a fragment. The
transform is therefore not a preprocessing detail — it is where the learning
signal comes from.

Two departures from the published recipe, both deliberate.

**No colour jitter or grayscale.** ``data/transforms.py`` records that hue
perturbation degrades the tissue-colour cue CVS criterion C2 depends on, and C2
is the highest-scoring criterion in all eight SAGES arms measured so far, so
this is a live risk rather than a hypothetical one. The reference recipe applies
ColorJitter with hue 0.05 at p=0.8 plus RandomGrayscale at p=0.2, which on
natural images makes the model invariant to lighting; on surgical video, tissue
colour is diagnostic. ``colour_jitter=True`` restores the reference behaviour
for a separate arm.

**Solarisation is also dropped.** It inverts pixels above a threshold, which on
laparoscopic imagery produces something no surgeon would recognise and nothing
the encoder will meet at evaluation.

Retained from the reference: two global views at scale (0.4, 1.0), local views
at (0.05, 0.4), horizontal flip, and asymmetric Gaussian blur — applied always
to the first global view, at p=0.1 to the second, and p=0.5 to locals. That
asymmetry matters: identical augmentation on both global views would let the
student match the teacher by matching the augmentation rather than the content.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class MultiCropTransform:
    """Produce ``2 + num_local`` views of one image.

    Returns a list, not a tensor: global and local views differ in resolution
    and cannot be stacked. ``multi_crop_collate`` batches them by size.

    Parameters
    ----------
    global_size, local_size:
        Resolutions. 224 and 96 in the reference recipe.
    num_local:
        Local views per image. Eight in the reference; the memory sweep shows
        that fits at batch 64 on a 32 GiB V100 (23.06 GiB), so there is no need
        to reduce it here.
    global_scale, local_scale:
        Area fractions for ``RandomResizedCrop``. The ranges are disjoint --
        (0.4, 1.0) against (0.05, 0.4) -- so a local view is always a strictly
        smaller region than a global one. Overlapping them would weaken the
        local-to-global asymmetry the objective depends on.
    colour_jitter:
        Restore the reference colour augmentation. Off by default; see the
        module docstring.
    """

    def __init__(
        self,
        *,
        global_size: int = 224,
        local_size: int = 96,
        num_local: int = 8,
        global_scale: tuple[float, float] = (0.4, 1.0),
        local_scale: tuple[float, float] = (0.05, 0.4),
        colour_jitter: bool = False,
        mean: Sequence[float] = IMAGENET_MEAN,
        std: Sequence[float] = IMAGENET_STD,
        interpolation: str = "bicubic",
    ) -> None:
        if num_local < 0:
            raise ValueError(f"num_local must be non-negative, got {num_local}")
        if not global_scale[0] < global_scale[1] <= 1.0:
            raise ValueError(f"global_scale must be increasing within (0, 1], got {global_scale}")
        if not local_scale[0] < local_scale[1] <= 1.0:
            raise ValueError(f"local_scale must be increasing within (0, 1], got {local_scale}")
        if local_scale[1] > global_scale[0]:
            raise ValueError(
                f"local_scale {local_scale} overlaps global_scale {global_scale}. "
                f"Local views must cover strictly smaller regions, or the "
                f"local-to-global correspondence the objective trains on is "
                f"weakened."
            )

        self.global_size = global_size
        self.local_size = local_size
        self.num_local = num_local
        self.colour_jitter = colour_jitter

        interp = {
            "bicubic": InterpolationMode.BICUBIC,
            "bilinear": InterpolationMode.BILINEAR,
        }[interpolation.lower()]

        flip = [T.RandomHorizontalFlip(p=0.5)]
        if colour_jitter:
            flip += [
                T.RandomApply(
                    [T.ColorJitter(brightness=0.4, contrast=0.4,
                                   saturation=0.2, hue=0.1)],
                    p=0.8,
                ),
                T.RandomGrayscale(p=0.2),
            ]

        normalise = [T.ToTensor(), T.Normalize(mean=list(mean), std=list(std))]

        def blur(p: float) -> list:
            # Kernel 23 at 224 px, scaled for local views: a fixed kernel would
            # blur a 96 px crop far more aggressively than a 224 px one.
            k = 23 if p >= 0 else 23
            return [T.RandomApply([T.GaussianBlur(kernel_size=k, sigma=(0.1, 2.0))], p=p)]

        def local_blur(p: float) -> list:
            k = max(3, (self.local_size // 10) * 2 + 1)
            return [T.RandomApply([T.GaussianBlur(kernel_size=k, sigma=(0.1, 2.0))], p=p)]

        crop_global = lambda: T.RandomResizedCrop(  # noqa: E731
            global_size, scale=global_scale, interpolation=interp, antialias=True
        )

        # Blur applied always to the first global view and at p=0.1 to the
        # second. Symmetric augmentation would let the student match the teacher
        # by matching the augmentation rather than the content.
        self.global_1 = T.Compose([crop_global(), *flip, *blur(1.0), *normalise])
        self.global_2 = T.Compose([crop_global(), *flip, *blur(0.1), *normalise])
        self.local = T.Compose([
            T.RandomResizedCrop(local_size, scale=local_scale,
                                interpolation=interp, antialias=True),
            *flip, *local_blur(0.5), *normalise,
        ])

    def __call__(self, image: Image.Image) -> list:
        views = [self.global_1(image), self.global_2(image)]
        views += [self.local(image) for _ in range(self.num_local)]
        return views

    @property
    def num_views(self) -> int:
        return 2 + self.num_local

    def describe(self) -> dict[str, Any]:
        """Recorded with a run, so the augmentation is reconstructable."""
        return {
            "global_size": self.global_size,
            "local_size": self.local_size,
            "num_local": self.num_local,
            "num_views": self.num_views,
            "colour_jitter": self.colour_jitter,
            "note": (
                "Colour jitter and grayscale are off by default: hue "
                "perturbation degrades the tissue-colour cue CVS criterion C2 "
                "depends on. Solarisation is omitted entirely."
            ),
        }

    def __repr__(self) -> str:
        return (
            f"MultiCropTransform({self.num_views} views: 2 x {self.global_size} "
            f"+ {self.num_local} x {self.local_size}, "
            f"colour_jitter={self.colour_jitter})"
        )


def multi_crop_collate(batch: list[list]) -> list:
    """Group a batch of view-lists into one tensor per view index.

    ``batch`` is ``[sample][view]``; the result is ``[view][sample]`` stacked,
    so ``out[0]`` is the first global view for every sample. Views of the same
    index share a resolution and stack cleanly; views of different indices may
    not, which is why this returns a list rather than a tensor.
    """
    import torch

    if not batch:
        raise ValueError("Empty batch.")
    num_views = len(batch[0])
    if any(len(sample) != num_views for sample in batch):
        raise ValueError(
            f"Inconsistent view counts in batch: "
            f"{sorted({len(s) for s in batch})}. Every sample must produce the "
            f"same number of views."
        )
    return [torch.stack([sample[i] for sample in batch]) for i in range(num_views)]


__all__ = ["MultiCropTransform", "multi_crop_collate"]
