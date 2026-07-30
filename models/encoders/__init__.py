"""Encoder registry.

Encoders are constructed by name so that experiment scripts, Slurm jobs, and
cache manifests refer to a single canonical string rather than an import path.
Registration is lazy: a checkpoint's dependencies are imported only when that
encoder is actually built, so a missing optional dependency does not break
unrelated experiments.
"""

from __future__ import annotations

from typing import Callable

from models.encoders.base_encoder import (
    BaseEncoder,
    EncoderOutput,
    Modality,
    PreprocessSpec,
    TokenLayout,
)


_REGISTRY: dict[str, Callable[..., BaseEncoder]] = {}


def register_encoder(name: str) -> Callable[[Callable[..., BaseEncoder]], Callable[..., BaseEncoder]]:
    """Decorator registering an encoder factory under a canonical name."""

    def decorator(factory: Callable[..., BaseEncoder]) -> Callable[..., BaseEncoder]:
        key = name.lower()
        if key in _REGISTRY:
            raise ValueError(f"Encoder {key!r} is already registered.")
        _REGISTRY[key] = factory
        return factory

    return decorator


def build_encoder(name: str, **kwargs) -> BaseEncoder:
    key = name.lower()
    if key not in _REGISTRY:
        raise KeyError(
            f"Unknown encoder {name!r}. Available: {sorted(_REGISTRY)}"
        )
    encoder = _REGISTRY[key](**kwargs)
    if not isinstance(encoder, BaseEncoder):
        raise TypeError(
            f"Factory for {name!r} returned {type(encoder).__name__}, which does "
            f"not implement BaseEncoder."
        )
    return encoder


def available_encoders() -> list[str]:
    return sorted(_REGISTRY)


__all__ = [
    "BaseEncoder",
    "EncoderOutput",
    "Modality",
    "PreprocessSpec",
    "TokenLayout",
    "available_encoders",
    "build_encoder",
    "register_encoder",
]