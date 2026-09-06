"""ViT-MAE encoder (masked pixel reconstruction, image level).

CRITICAL: ``ViTMAEModel`` performs random masking inside ``forward``, with
``config.mask_ratio`` defaulting to 0.75. At 224 px this returns 50 of the 197
tokens, and a *different* 50 on every call. Nothing raises; the output is a
well-formed tensor of the wrong length, non-deterministically.

That behaviour is correct for pretraining and catastrophic for feature
extraction, where it would silently produce a cache that is neither complete nor
reproducible. This wrapper forces ``mask_ratio = 0.0`` and asserts the resulting
token count against the declared layout, so the failure mode cannot recur.
"""

from __future__ import annotations

import torch

from pathlib import Path
from typing import Any

from models.encoders import register_encoder
from models.encoders.base_encoder import BaseEncoder, EncoderOutput, PreprocessSpec, TokenLayout
from models.encoders.hf_common import (
    cfg_get,
    image_spec,
    require_transformers,
    spatial_grid,
    vit_dims,
)


class MAEEncoder(BaseEncoder):
    modality = "image"

    CHECKPOINTS = {
        "base": "facebook/vit-mae-base",
        "large": "facebook/vit-mae-large",
        "huge": "facebook/vit-mae-huge",
    }

    def __init__(
        self,
        variant: str = "base",
        *,
        model_name: str | None = None,
        model=None,
        random_init: bool = False,
        adapted_checkpoint: str | Path | None = None,
        freeze: bool = True,
    ) -> None:
        super().__init__(freeze=freeze)

        # pretrain_mae.py exports the same payload shape as pretrain_dino.py --
        # the encoder under "model", with step and config alongside -- so the
        # loader is shared. What differs is that it saves ``model.vit``, the
        # inner ViT of a ViTMAEForPreTraining, whereas extraction wants a
        # ViTMAEModel; the loader's missing-key check will refuse the checkpoint
        # rather than load a partial one if the two disagree.
        self._adaptation: dict[str, Any] | None = None
        if adapted_checkpoint is not None:
            if model is not None or random_init:
                raise ValueError(
                    "adapted_checkpoint cannot be combined with model= or "
                    "random_init=True."
                )
            require_transformers()
            from transformers import ViTMAEModel

            from models.encoders.dinov2_encoder import load_adapted_checkpoint

            model, base, self._adaptation = load_adapted_checkpoint(
                adapted_checkpoint,
                base_checkpoint=model_name,
                fallback_base=self.CHECKPOINTS.get(variant),
                model_cls=ViTMAEModel,
            )
            model_name = f"{base}+adapted:{Path(adapted_checkpoint).name}"

        if model is None and random_init:
            require_transformers()
            from transformers import ViTMAEConfig, ViTMAEModel

            model = ViTMAEModel(
                ViTMAEConfig(image_size=224, patch_size=16, mask_ratio=0.0, **vit_dims(variant))
            )
            model_name = f"random-init-{variant}"

        if model is None:
            require_transformers()
            from transformers import ViTMAEConfig, ViTMAEModel

            if model_name is None:
                if variant not in self.CHECKPOINTS:
                    raise ValueError(
                        f"Unknown variant {variant!r}. Available: {sorted(self.CHECKPOINTS)}"
                    )
                model_name = self.CHECKPOINTS[variant]
            config = ViTMAEConfig.from_pretrained(model_name)
            config.mask_ratio = 0.0
            model = ViTMAEModel.from_pretrained(model_name, config=config)

        # Applies to caller-supplied models too: the guarantee must hold
        # regardless of how the model arrived.
        if getattr(model.config, "mask_ratio", 0.0) != 0.0:
            model.config.mask_ratio = 0.0
        if hasattr(model.embeddings, "config"):
            model.embeddings.config.mask_ratio = 0.0

        self.model = model
        self._model_name = model_name or "from-config"

        cfg = model.config
        image_size = int(cfg_get(cfg, "image_size"))
        patch_size = int(cfg_get(cfg, "patch_size"))
        h, w = spatial_grid(image_size, patch_size)

        self._spec = image_spec(image_size)
        self._layout = TokenLayout(
            grid=(h, w), dim=int(cfg_get(cfg, "hidden_size")), num_prefix_tokens=1
        )
        self._finalise_init()

    def describe(self) -> dict:
        """Recorded in the cache manifest, so that an adapted cache is
        distinguishable from a baseline one after the fact."""
        record = super().describe()
        if getattr(self, "_adaptation", None) is not None:
            record["adaptation"] = self._adaptation
        return record

    @property
    def preprocess_spec(self) -> PreprocessSpec:
        return self._spec

    @property
    def token_layout(self) -> TokenLayout:
        return self._layout

    @property
    def checkpoint_id(self) -> str:
        return self._model_name

    def _monotonic_noise(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Noise whose argsort is the identity permutation.

        ``ViTMAEEmbeddings.random_masking`` shuffles tokens by ``argsort(noise)``
        and keeps the leading ``len_keep``. Setting ``mask_ratio = 0`` makes it
        keep all of them, but the *shuffle still happens*: patch tokens are
        returned in a random order that changes on every call.

        Supplying strictly increasing noise makes the argsort an identity
        permutation, so tokens are returned in raster order, bit-exactly
        reproducibly. Restoring order afterwards via the returned
        ``ids_restore`` also works, but is only accurate to floating-point
        tolerance because the permuted attention reduction reorders the
        arithmetic.
        """
        n = self._layout.num_tokens
        return torch.arange(n, dtype=torch.float32, device=device).div(n).expand(batch_size, n)

    def _forward_tokens(
        self,
        x: torch.Tensor,
        *,
        layer_depths: tuple[float, ...] | None = None,
    ) -> EncoderOutput:
        # The monotonic noise is what keeps ViT-MAE's masking from permuting
        # the patch order, so it is passed on every path including this one.
        # output_hidden_states does not change the masking, but omitting the
        # noise would.
        out = self.model(
            pixel_values=x,
            noise=self._monotonic_noise(x.shape[0], x.device),
            output_hidden_states=layer_depths is not None,
        )
        if layer_depths is not None:
            from models.encoders.base_encoder import resolve_layer_indices

            n_prefix = self._layout.num_prefix_tokens
            indices = resolve_layer_indices(
                layer_depths, self.model.config.num_hidden_layers
            )
            # Two adjustments, both needed for a depth comparison to compare
            # depth alone. The CLS token is stripped from every selected layer,
            # which the base class requires so the spatial layout is identical
            # along the layer axis. And the terminal LayerNorm is applied:
            # hidden_states holds each block's raw output while last_hidden_state
            # is the final block after that norm, so without it depth 1.0 would
            # not reproduce the tensor every other experiment cached and the
            # depths would differ in scale as well as in depth. Applying a norm
            # fitted for block 12 to block 3 is not neutral, but the alternative
            # confounds depth with feature scale, which is worse.
            norm = getattr(self.model, "layernorm", None) or getattr(
                self.model, "norm", None
            )
            if norm is None:
                raise RuntimeError(
                    "No terminal LayerNorm found on the ViT-MAE model, so the "
                    "selected layers cannot be brought onto the same scale as "
                    "last_hidden_state."
                )
            selected = torch.stack(
                [norm(out.hidden_states[i])[:, n_prefix:, :] for i in indices],
                dim=1,
            )
        else:
            selected = None
        hidden = out.last_hidden_state

        expected = self._layout.num_tokens + self._layout.num_prefix_tokens
        if hidden.shape[1] != expected:
            raise RuntimeError(
                f"ViT-MAE returned {hidden.shape[1]} tokens, expected {expected}. "
                f"Masking is still active (mask_ratio="
                f"{getattr(self.model.config, 'mask_ratio', '?')}); extracted "
                f"features would be incomplete and non-deterministic."
            )

        ids_restore = getattr(out, "ids_restore", None)
        if ids_restore is not None:
            reference = torch.arange(self._layout.num_tokens, device=ids_restore.device)
            if not torch.equal(ids_restore[0], reference):
                raise RuntimeError(
                    "ViT-MAE permuted its patch tokens despite monotonic noise. "
                    "Cached features would not correspond to the declared "
                    f"{self._layout.grid} grid. The upstream masking "
                    "implementation has changed and this wrapper needs updating."
                )

        return EncoderOutput(tokens=hidden[:, 1:, :], prefix=hidden[:, :1, :],
                             hidden_states=selected)


@register_encoder("mae_b")
def _mae_b(**kwargs) -> MAEEncoder:
    return MAEEncoder(variant="base", **kwargs)


@register_encoder("mae_l")
def _mae_l(**kwargs) -> MAEEncoder:
    return MAEEncoder(variant="large", **kwargs)


__all__ = ["MAEEncoder"]