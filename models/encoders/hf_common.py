"""Shared helpers for HuggingFace-backed encoders.

Geometry is derived from each checkpoint's own config rather than hard-coded, so
that swapping ViT-B for ViT-L, or 224 px for 256 px, does not require editing
the wrapper.
"""

from __future__ import annotations

from typing import Any

from models.encoders.base_encoder import PreprocessSpec

# All four checkpoint families in this project normalise with ImageNet
# statistics. Verify against the checkpoint's own preprocessor_config.json on
# first download; if a family diverges, pass mean/std explicitly rather than
# editing these constants.
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


def require_transformers() -> None:
    try:
        import transformers  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "transformers is required for this encoder. "
            "Install it with `pip install -r requirements.txt`."
        ) from exc


def cfg_get(config: Any, *names: str, default: Any = None, required: bool = True) -> Any:
    """Read the first attribute present on a config.

    HuggingFace config field names differ across model families for the same
    concept (``image_size`` versus ``crop_size``, ``num_frames`` versus
    ``frames_per_clip``). Rather than branch at each call site, try the known
    aliases and fail loudly if none is present, since a silently wrong geometry
    produces a cache that looks valid and is not.
    """
    for name in names:
        if hasattr(config, name):
            value = getattr(config, name)
            if value is not None:
                return value
    if required and default is None:
        raise AttributeError(
            f"{type(config).__name__} exposes none of {names}. The checkpoint "
            f"geometry cannot be determined; the wrapper needs updating for "
            f"this transformers version."
        )
    return default


def spatial_grid(image_size: int, patch_size: int) -> tuple[int, int]:
    if image_size % patch_size != 0:
        raise ValueError(
            f"image_size {image_size} is not divisible by patch_size {patch_size}."
        )
    side = image_size // patch_size
    return side, side


def temporal_grid(num_frames: int, tubelet_size: int) -> int:
    if num_frames % tubelet_size != 0:
        raise ValueError(
            f"num_frames {num_frames} is not divisible by tubelet_size {tubelet_size}."
        )
    return num_frames // tubelet_size


def image_spec(image_size: int, mean=IMAGENET_MEAN, std=IMAGENET_STD) -> PreprocessSpec:
    return PreprocessSpec(image_size=image_size, mean=mean, std=std)


def video_spec(
    image_size: int,
    num_frames: int,
    frame_stride: int,
    mean=IMAGENET_MEAN,
    std=IMAGENET_STD,
) -> PreprocessSpec:
    return PreprocessSpec(
        image_size=image_size,
        mean=mean,
        std=std,
        num_frames=num_frames,
        frame_stride=frame_stride,
    )


__all__ = [
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "cfg_get",
    "image_spec",
    "require_transformers",
    "spatial_grid",
    "temporal_grid",
    "video_spec",
]