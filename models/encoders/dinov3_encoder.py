"""DINOv3 ViT encoder (self-distillation, image level).

Used both as a frozen baseline and, since a 32 GiB V100 admits the full 2+8
multi-crop recipe at batch 64, as an adapted arm. ``adapted_checkpoint`` loads a
teacher backbone exported by ``train/pretrain_dino.py`` through the loader
shared with DINOv2; the two families are adapted by the same trainer and differ
only in the model class the state dict is loaded into.

DINOv3 prepends a CLS token *and* register tokens. Register tokens carry no
spatial position, so treating them as patches would corrupt any attention map
over the grid. The register count is read from the loaded checkpoint config
rather than the class default, which is zero and does not match the released
weights.

The weights are gated, which needs two separate things before first use:

1. Access granted on the model page, while signed in to the account whose token
   you will use. Without it the download fails 403 even when authenticated.
2. ``hf auth login`` in the shell. ``huggingface-cli login`` is a no-op stub
   from huggingface_hub 1.x onward: it prints a deprecation hint, exits zero,
   and authenticates nothing, so the next call fails 401 as though no attempt
   had been made.

Export ``HF_HOME`` *before* logging in if the token must be visible to a batch
job. The token is written under ``$HF_HOME``, so logging in with the variable
unset stores it in ``~/.cache/huggingface``, where a job that points ``HF_HOME``
at project storage will not find it.
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


class DINOv3Encoder(BaseEncoder):
    modality = "image"

    CHECKPOINTS = {
        "small": "facebook/dinov3-vits16-pretrain-lvd1689m",
        "base": "facebook/dinov3-vitb16-pretrain-lvd1689m",
        "large": "facebook/dinov3-vitl16-pretrain-lvd1689m",
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

        # Continued pretraining is performed by the same trainer that adapts
        # DINOv2, and exports the same payload -- the teacher backbone under
        # "model" -- so the loader is shared and only the model class differs.
        self._adaptation: dict[str, Any] | None = None
        if adapted_checkpoint is not None:
            if model is not None or random_init:
                raise ValueError(
                    "adapted_checkpoint cannot be combined with model= or "
                    "random_init=True."
                )
            require_transformers()
            from transformers import DINOv3ViTModel

            from models.encoders.dinov2_encoder import load_adapted_checkpoint

            model, base, self._adaptation = load_adapted_checkpoint(
                adapted_checkpoint,
                base_checkpoint=model_name,
                fallback_base=self.CHECKPOINTS.get(variant),
                model_cls=DINOv3ViTModel,
            )
            model_name = f"{base}+adapted:{Path(adapted_checkpoint).name}"

        if model is None and random_init:
            require_transformers()
            from transformers import DINOv3ViTConfig, DINOv3ViTModel

            model = DINOv3ViTModel(
                DINOv3ViTConfig(
                    image_size=224, patch_size=16, num_register_tokens=4, **vit_dims(variant)
                )
            )
            model_name = f"random-init-{variant}"

        if model is None:
            require_transformers()
            from transformers import DINOv3ViTModel

            if model_name is None:
                if variant not in self.CHECKPOINTS:
                    raise ValueError(
                        f"Unknown variant {variant!r}. Available: {sorted(self.CHECKPOINTS)}"
                    )
                model_name = self.CHECKPOINTS[variant]
            model = DINOv3ViTModel.from_pretrained(model_name)

        self.model = model
        self._model_name = model_name or "from-config"

        cfg = model.config
        image_size = int(cfg_get(cfg, "image_size"))
        patch_size = int(cfg_get(cfg, "patch_size"))
        h, w = spatial_grid(image_size, patch_size)

        num_registers = int(cfg_get(cfg, "num_register_tokens", default=0, required=False) or 0)
        self._num_registers = num_registers

        self._spec = image_spec(image_size)
        self._layout = TokenLayout(
            grid=(h, w),
            dim=int(cfg_get(cfg, "hidden_size")),
            num_prefix_tokens=1 + num_registers,
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

    @property
    def num_register_tokens(self) -> int:
        return self._num_registers

    def _forward_tokens(
        self,
        x: torch.Tensor,
        *,
        layer_depths: tuple[float, ...] | None = None,
    ) -> EncoderOutput:
        if layer_depths is not None:
            from models.encoders.base_encoder import resolve_layer_indices

            out = self.model(pixel_values=x, output_hidden_states=True)
            n_prefix = self._layout.num_prefix_tokens
            indices = resolve_layer_indices(
                layer_depths, self.model.config.num_hidden_layers
            )
            # Two adjustments, both required for a depth comparison to be a
            # comparison of depth alone.
            #
            # Prefix stripping: unlike VideoMAE, every DINOv3 hidden state
            # carries a CLS token and four registers, and the base class
            # requires them removed from each selected layer so the spatial
            # layout is identical along the layer axis.
            #
            # Final normalisation: ``hidden_states`` holds each block's raw
            # output, whereas ``last_hidden_state`` is the final block's output
            # after the model's terminal LayerNorm. Without applying it, depth
            # 1.0 would not reproduce the tensor every other experiment in this
            # project cached, and the depths would differ in scale as well as in
            # depth. Applying a normalisation fitted for block 12 to block 3 is
            # not neutral, but the alternative confounds depth with feature
            # scale, which is worse; the choice is recorded in the manifest by
            # way of layer_depths being set.
            norm = getattr(self.model, "layernorm", None) or getattr(
                self.model, "norm", None
            )
            if norm is None:
                raise RuntimeError(
                    "No terminal LayerNorm found on the DINOv3 model, so the "
                    "selected layers cannot be brought onto the same scale as "
                    "last_hidden_state."
                )
            selected = torch.stack(
                [norm(out.hidden_states[i])[:, n_prefix:, :] for i in indices],
                dim=1,
            )
            hidden = out.last_hidden_state
            expected = self._layout.num_tokens + n_prefix
            if hidden.shape[1] != expected:
                raise RuntimeError(
                    f"DINOv3 returned {hidden.shape[1]} tokens, expected {expected}."
                )
            return EncoderOutput(
                tokens=hidden[:, n_prefix:, :],
                prefix=hidden[:, :n_prefix, :],
                hidden_states=selected,
            )

        hidden = self.model(pixel_values=x).last_hidden_state
        n_prefix = self._layout.num_prefix_tokens
        expected = self._layout.num_tokens + n_prefix
        if hidden.shape[1] != expected:
            raise RuntimeError(
                f"DINOv3 returned {hidden.shape[1]} tokens, expected {expected} "
                f"({n_prefix} prefix + {self._layout.num_tokens} patch). The "
                f"register-token count from the config does not match the "
                f"checkpoint; patch tokens would be misaligned."
            )
        return EncoderOutput(tokens=hidden[:, n_prefix:, :], prefix=hidden[:, :n_prefix, :])


@register_encoder("dinov3_s")
def _dinov3_s(**kwargs) -> DINOv3Encoder:
    return DINOv3Encoder(variant="small", **kwargs)


@register_encoder("dinov3_b")
def _dinov3_b(**kwargs) -> DINOv3Encoder:
    return DINOv3Encoder(variant="base", **kwargs)


__all__ = ["DINOv3Encoder"]