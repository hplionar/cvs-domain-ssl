"""DINOv3 ViT encoder (self-distillation, image level).

Included as a frozen baseline in E1. Adaptation is out of scope: DINO-style
self-distillation depends on batch statistics for centering and sharpening, so
gradient accumulation does not substitute for a large true batch on a single
V100. See docs/implementation_plan.md §10.

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
        freeze: bool = True,
    ) -> None:
        super().__init__(freeze=freeze)

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

    def _forward_tokens(self, x: torch.Tensor) -> EncoderOutput:
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