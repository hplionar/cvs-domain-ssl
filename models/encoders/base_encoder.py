"""Abstract encoder interface for CVS domain-adaptive SSL experiments.

Every encoder in this project returns a **token grid** of shape ``[B, N, D]``
rather than a pooled vector. This is a deliberate architectural constraint:

- Pooling is a lossy, irreversible reduction. Performing it inside the encoder
  would prevent attentive (MIL) aggregation over patches or tubelets, and would
  make the cached features unusable for any head other than the one they were
  extracted for.
- Token grids are a strict superset. A pooled vector is recoverable from a token
  grid in microseconds; the reverse is impossible.

Pooling therefore belongs in the head (``models/heads/``), never here.

Determinism
-----------
The cached-feature architecture requires that ``extract`` be a deterministic
function of its input. Encoders are frozen and held in eval mode by default, and
``train()`` is overridden so that a parent module switching to training mode
cannot silently re-enable dropout or batch-norm updates inside a frozen encoder.

Input conventions
-----------------
- Image encoders accept ``[B, C, H, W]``.
- Video encoders accept ``[B, T, C, H, W]``.

``[B, T, C, H, W]`` is chosen because it is what ``torch.stack`` over a list of
per-frame ``[C, H, W]`` tensors produces, which is what the clip datasets in
``data/`` already emit. Wrappers permute internally if their upstream
implementation expects ``[B, C, T, H, W]``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from math import prod
from typing import Any, ClassVar, Literal, NamedTuple

import torch
import torch.nn as nn


Modality = Literal["image", "video"]


class EncoderOutput(NamedTuple):
    """Structured encoder output.

    Attributes
    ----------
    tokens:
        Patch or tubelet tokens, shape ``[B, N, D]``. Prefix tokens are already
        removed, so ``N`` equals the product of ``TokenLayout.grid``.
    prefix:
        Prefix tokens (CLS, and register tokens where present), shape
        ``[B, P, D]``, or ``None`` if the architecture has none.

        Retained rather than discarded because several published linear-probe
        protocols concatenate the CLS token with mean-pooled patch tokens. Note
        that DINOv3 register tokens are *not* spatial and must never be treated
	as patches; keeping them here, separated, prevents that error.
    

    hidden_states:
        Patch tokens from selected intermediate layers, shape
        ``[B, L, N, D]`` where ``L`` is the number of layers requested, or
        ``None`` when they were not requested. Prefix tokens are removed from
        each layer exactly as they are from ``tokens``, so the spatial layout is
        identical across the layer axis.

        Requested because feature extraction otherwise keeps only the final
        layer. If different CVS criteria are resolved at different levels of
        abstraction -- C2 (tissue cleared) plausibly earlier than C1 (counting
        two structures) -- a last-layer probe discards the evidence.
    """

    tokens: torch.Tensor
    prefix: torch.Tensor | None = None
    hidden_states: torch.Tensor | None = None


@dataclass(frozen=True)
class PreprocessSpec:
    """Preprocessing required by a specific checkpoint.

    Carried by the encoder rather than assumed by the caller. Checkpoints differ
    in patch size, input resolution, and normalisation statistics; hard-coding a
    single transform across arms would introduce exactly the kind of protocol
    asymmetry that invalidates an adaptation-gain comparison.
    """

    image_size: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    interpolation: str = "bicubic"
    num_frames: int | None = None
    frame_stride: int | None = None

    def __post_init__(self) -> None:
        if self.image_size <= 0:
            raise ValueError(f"image_size must be positive, got {self.image_size}")
        if len(self.mean) != 3 or len(self.std) != 3:
            raise ValueError("mean and std must each have three elements.")
        if any(s <= 0 for s in self.std):
            raise ValueError(f"std entries must be positive, got {self.std}")
        if (self.num_frames is None) != (self.frame_stride is None):
            raise ValueError(
                "num_frames and frame_stride must be specified together or "
                "both omitted."
            )
        if self.num_frames is not None and self.num_frames <= 0:
            raise ValueError(f"num_frames must be positive, got {self.num_frames}")


@dataclass(frozen=True)
class TokenLayout:
    """Spatial or spatiotemporal structure of the returned token grid.

    Required so that a head can reshape ``[B, N, D]`` back into its grid without
    guessing, and so that the extraction script can record the layout alongside
    the cache.
    """

    grid: tuple[int, ...]
    dim: int
    num_prefix_tokens: int = 0

    def __post_init__(self) -> None:
        if len(self.grid) not in (2, 3):
            raise ValueError(
                f"grid must be (H', W') or (T', H', W'), got {self.grid}"
            )
        if any(g <= 0 for g in self.grid):
            raise ValueError(f"grid entries must be positive, got {self.grid}")
        if self.dim <= 0:
            raise ValueError(f"dim must be positive, got {self.dim}")
        if self.num_prefix_tokens < 0:
            raise ValueError(
                f"num_prefix_tokens must be non-negative, got {self.num_prefix_tokens}"
            )

    @property
    def num_tokens(self) -> int:
        """Number of patch or tubelet tokens, excluding prefix tokens."""
        return prod(self.grid)

    @property
    def is_spatiotemporal(self) -> bool:
        return len(self.grid) == 3

    def bytes_per_sample(self, dtype_size: int = 2) -> int:
        """Cache footprint of one sample's token grid, default fp16."""
        return self.num_tokens * self.dim * dtype_size


def resolve_layer_indices(
    relative_depths: tuple[float, ...],
    num_layers: int,
) -> tuple[int, ...]:
    """Map relative depths in (0, 1] to indices into a hidden_states tuple.

    HuggingFace returns ``num_layers + 1`` tensors, index 0 being the embedding
    output before any block. Depth ``d`` maps to index ``round(d * num_layers)``,
    so 1.0 is the final block and 0.25 of a 12-block model is block 3.

    Index 0 is unreachable by construction: the embedding output precedes every
    transformer block and is close to a linear projection of pixels, so a
    "shallow layers do not help" conclusion should not be able to rest on it.
    """
    if not relative_depths:
        raise ValueError("relative_depths must not be empty.")
    if any(not 0.0 < d <= 1.0 for d in relative_depths):
        raise ValueError(
            f"relative depths must lie in (0, 1], got {relative_depths}. "
            f"Depth 0 is the embedding output, which is deliberately excluded."
        )
    indices = tuple(max(1, round(d * num_layers)) for d in relative_depths)
    if len(set(indices)) != len(indices):
        raise ValueError(
            f"Relative depths {relative_depths} collapse to duplicate layer "
            f"indices {indices} for a {num_layers}-layer model. Request fewer "
            f"depths or space them further apart."
        )
    return indices


class BaseEncoder(nn.Module, ABC):
    """Abstract frozen feature extractor returning token grids.

    Subclasses implement ``_forward_tokens`` and declare ``preprocess_spec`` and
    ``token_layout``. All shape validation, freezing, and eval-mode enforcement
    is handled here so that it cannot be inconsistently reimplemented per
    encoder.
    """

    modality: ClassVar[Modality]

    def __init__(self, *, freeze: bool = True) -> None:
        super().__init__()
        if not hasattr(type(self), "modality"):
            raise TypeError(
                f"{type(self).__name__} must declare a class-level "
                f"'modality' of 'image' or 'video'."
            )
        self._frozen = False
        self._freeze_requested = freeze

    def _finalise_init(self) -> None:
        """Call at the end of a subclass ``__init__``, after weights are loaded."""
        if self._freeze_requested:
            self.freeze()

    # -- contract ---------------------------------------------------------

    @property
    @abstractmethod
    def preprocess_spec(self) -> PreprocessSpec:
        """Preprocessing this checkpoint requires."""

    @property
    @abstractmethod
    def token_layout(self) -> TokenLayout:
        """Structure of the returned token grid."""

    @abstractmethod
    def _forward_tokens(
        self,
        x: torch.Tensor,
        *,
        layer_depths: tuple[float, ...] | None = None,
    ) -> EncoderOutput:
        """Run the underlying model and return patch tokens with prefix separated.

        Implementations that support ``layer_depths`` must also strip prefix
        tokens from each returned layer, so that the spatial layout is identical
        along the layer axis.

        Subclasses must strip CLS and register tokens from ``tokens`` and place
        them in ``prefix``. Returning them concatenated is a contract violation
        and will be caught by ``forward``.
        """

    # -- derived ----------------------------------------------------------

    @property
    def feature_dim(self) -> int:
        return self.token_layout.dim

    @property
    def is_frozen(self) -> bool:
        return self._frozen

    @property
    def checkpoint_id(self) -> str:
        """Stable identifier recorded in the cache manifest for provenance."""
        return type(self).__name__

    def describe(self) -> dict[str, Any]:
        """Provenance record written alongside extracted features.

        Two runs whose ``describe()`` outputs differ did not use an identical
        protocol, and their adaptation gains are not comparable. This makes that
        checkable rather than assumed.
        """
        return {
            "class": type(self).__name__,
            "checkpoint_id": self.checkpoint_id,
            "modality": self.modality,
            "frozen": self._frozen,
            "preprocess": asdict(self.preprocess_spec),
            "token_layout": asdict(self.token_layout),
            "num_parameters": sum(p.numel() for p in self.parameters()),
        }

    # -- freezing ---------------------------------------------------------

    def freeze(self) -> "BaseEncoder":
        for param in self.parameters():
            param.requires_grad_(False)
        self._frozen = True
        super().train(False)
        return self

    def unfreeze(self) -> "BaseEncoder":
        for param in self.parameters():
            param.requires_grad_(True)
        self._frozen = False
        return self

    def train(self, mode: bool = True) -> "BaseEncoder":
        """Ignore training mode while frozen.

        A parent ``nn.Module`` calling ``.train()`` must not be able to
        re-enable stochastic behaviour inside a frozen encoder, since the cached
        features assume deterministic extraction.
        """
        if self._frozen:
            return super().train(False)
        return super().train(mode)

    # -- validation -------------------------------------------------------

    def _validate_input(self, x: torch.Tensor) -> None:
        expected_ndim = 4 if self.modality == "image" else 5
        if x.ndim != expected_ndim:
            shape_str = "[B, C, H, W]" if self.modality == "image" else "[B, T, C, H, W]"
            raise ValueError(
                f"{type(self).__name__} expects {shape_str} "
                f"({expected_ndim}D), got {tuple(x.shape)}."
            )

        channels = x.shape[1] if self.modality == "image" else x.shape[2]
        if channels != 3:
            raise ValueError(
                f"Expected 3 input channels, got {channels}. For video the "
                f"convention is [B, T, C, H, W]; a tensor shaped "
                f"[B, C, T, H, W] will trip this check."
            )

        spec = self.preprocess_spec
        height, width = x.shape[-2], x.shape[-1]
        if (height, width) != (spec.image_size, spec.image_size):
            raise ValueError(
                f"Expected {spec.image_size}x{spec.image_size} input, got "
                f"{height}x{width}. Build the transform from "
                f"encoder.preprocess_spec rather than a fixed constant."
            )

        if self.modality == "video" and spec.num_frames is not None:
            if x.shape[1] != spec.num_frames:
                raise ValueError(
                    f"Expected {spec.num_frames} frames, got {x.shape[1]}."
                )

    def _validate_output(self, out: EncoderOutput, batch_size: int) -> None:
        layout = self.token_layout
        tokens = out.tokens

        if tokens.ndim != 3:
            raise ValueError(
                f"{type(self).__name__}._forward_tokens must return tokens of "
                f"shape [B, N, D], got {tuple(tokens.shape)}."
            )
        if tokens.shape[0] != batch_size:
            raise ValueError(
                f"Batch size changed through the encoder: input {batch_size}, "
                f"output {tokens.shape[0]}."
            )
        if tokens.shape[1] != layout.num_tokens:
            raise ValueError(
                f"Token count {tokens.shape[1]} does not match declared layout "
                f"{layout.grid} ({layout.num_tokens} tokens). If the difference "
                f"equals {layout.num_prefix_tokens}, prefix tokens were not "
                f"stripped into EncoderOutput.prefix."
            )
        if tokens.shape[2] != layout.dim:
            raise ValueError(
                f"Token dimension {tokens.shape[2]} does not match declared "
                f"dim {layout.dim}."
            )

        if out.prefix is not None:
            if out.prefix.ndim != 3:
                raise ValueError(
                    f"prefix must have shape [B, P, D], got {tuple(out.prefix.shape)}."
                )
            if out.prefix.shape[0] != batch_size:
                raise ValueError("prefix batch size does not match input.")
            if out.prefix.shape[1] != layout.num_prefix_tokens:
                raise ValueError(
                    f"prefix has {out.prefix.shape[1]} tokens but layout "
                    f"declares {layout.num_prefix_tokens}."
                )
            if out.prefix.shape[2] != layout.dim:
                raise ValueError("prefix dimension does not match layout dim.")
        elif layout.num_prefix_tokens != 0:
            raise ValueError(
                f"Layout declares {layout.num_prefix_tokens} prefix tokens but "
                f"_forward_tokens returned prefix=None."
            )

    # -- entry points -----------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        *,
        layer_depths: tuple[float, ...] | None = None,
    ) -> EncoderOutput:
        """Run the encoder.

        ``layer_depths`` requests intermediate layers as relative depths in
        (0, 1]; see ``resolve_layer_indices``. When omitted, only the final
        layer is returned and behaviour is unchanged.
        """
        self._validate_input(x)
        if layer_depths is None:
            out = self._forward_tokens(x)
        else:
            out = self._forward_tokens(x, layer_depths=layer_depths)
        self._validate_output(out, batch_size=x.shape[0])
        if layer_depths is not None and out.hidden_states is None:
            raise RuntimeError(
                f"{type(self).__name__} ignored layer_depths and returned no "
                f"hidden states. Silently falling back to the final layer would "
                f"make a depth comparison meaningless."
            )
        return out

    @torch.no_grad()
    def extract(
        self,
        x: torch.Tensor,
        *,
        to_dtype: torch.dtype | None = torch.float16,
        layer_depths: tuple[float, ...] | None = None,
    ) -> EncoderOutput:
        """Deterministic extraction path used by ``scripts/extract_features.py``.

        Wraps ``forward`` in ``no_grad`` and optionally casts to the cache dtype.
        Raises if the encoder is not frozen, because caching the output of a
        trainable encoder would silently produce a stale cache.
        """
        if not self._frozen:
            raise RuntimeError(
                "extract() requires a frozen encoder. Call freeze() first, or "
                "use forward() if gradients are genuinely wanted."
            )
        was_training = self.training
        self.eval()
        try:
            out = self.forward(x, layer_depths=layer_depths)
        finally:
            if was_training:
                super().train(True)

        if to_dtype is None:
            return out
        return EncoderOutput(
            tokens=out.tokens.to(to_dtype),
            prefix=None if out.prefix is None else out.prefix.to(to_dtype),
            hidden_states=(None if out.hidden_states is None else out.hidden_states.to(to_dtype)),
        )

    def __repr__(self) -> str:
        layout = self.token_layout
        return (
            f"{type(self).__name__}(modality={self.modality!r}, "
            f"grid={layout.grid}, dim={layout.dim}, "
            f"prefix={layout.num_prefix_tokens}, frozen={self._frozen})"
        )


__all__ = [
    "BaseEncoder",
    "EncoderOutput",
    "Modality",
    "PreprocessSpec",
    "TokenLayout",
]
