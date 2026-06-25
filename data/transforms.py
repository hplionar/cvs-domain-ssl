"""Image transform builders for Endoscapes experiments.

The dataset returns PIL RGB images. These builders convert images to tensors
and apply the preprocessing used by the first frame-level baselines.
"""

from __future__ import annotations

from collections.abc import Callable

from torchvision import transforms
from torchvision.transforms import InterpolationMode


IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


def build_ssl_train_transform(
    image_size: int = 224,
    scale: tuple[float, float] = (0.2, 1.0),
    ratio: tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),
) -> Callable:
    """Build augmentation for self-supervised training frames.

    This uses all available frames and intentionally applies stronger image
    augmentation than the supervised transform.
    """

    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                image_size,
                scale=scale,
                ratio=ratio,
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.4,
                        contrast=0.4,
                        saturation=0.2,
                        hue=0.05,
                    )
                ],
                p=0.8,
            ),
            transforms.RandomGrayscale(p=0.1),
            transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_supervised_train_transform(
    image_size: int = 224,
    scale: tuple[float, float] = (0.7, 1.0),
    ratio: tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),
) -> Callable:
    """Build augmentation for supervised CVS keyframe training."""

    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                image_size,
                scale=scale,
                ratio=ratio,
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.1,
                hue=0.03,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_eval_transform(
    image_size: int = 224,
    resize_size: int = 256,
) -> Callable:
    """Build deterministic preprocessing for validation and test frames."""

    return transforms.Compose(
        [
            transforms.Resize(
                resize_size,
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_transform(
    mode: str,
    split: str,
    image_size: int = 224,
) -> Callable:
    """Dispatch to the appropriate transform for a dataset mode and split."""

    mode = mode.lower()
    split = split.lower()

    if split in {"val", "test"}:
        return build_eval_transform(image_size=image_size)

    if split != "train":
        raise ValueError(f"Unknown split: {split!r}. Expected train, val, or test.")

    if mode == "ssl":
        return build_ssl_train_transform(image_size=image_size)
    if mode == "supervised":
        return build_supervised_train_transform(image_size=image_size)

    raise ValueError(f"Unknown mode: {mode!r}. Expected ssl or supervised.")


__all__ = [
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "build_eval_transform",
    "build_ssl_train_transform",
    "build_supervised_train_transform",
    "build_transform",
]
