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


#: Standard ViT dimensions, used to build randomly initialised models with real
#: geometry for smoke-testing throughput and memory without downloading weights.
VIT_DIMS: dict[str, dict[str, int]] = {
    "small": {"hidden_size": 384, "num_hidden_layers": 12, "num_attention_heads": 6, "intermediate_size": 1536},
    "base": {"hidden_size": 768, "num_hidden_layers": 12, "num_attention_heads": 12, "intermediate_size": 3072},
    "large": {"hidden_size": 1024, "num_hidden_layers": 24, "num_attention_heads": 16, "intermediate_size": 4096},
    "huge": {"hidden_size": 1280, "num_hidden_layers": 32, "num_attention_heads": 16, "intermediate_size": 5120},
}


def vit_dims(variant: str) -> dict[str, int]:
    if variant not in VIT_DIMS:
        raise ValueError(f"No standard dimensions for variant {variant!r}.")
    return dict(VIT_DIMS[variant])


__all__ += ["VIT_DIMS", "vit_dims"]