"""V-JEPA 2 encoder (masked latent prediction, video level).

Represents the *latent prediction* arm of the E2 comparison.

Two details that would otherwise corrupt extraction:

1. ``VJEPA2Model.forward`` runs the predictor unless ``skip_predictor=True``.
   The predictor is the SSL head, not part of the representation, and running it
   wastes compute while returning an output object whose extra fields invite
   the wrong tensor being cached.
2. The ViT-B checkpoint is a *distilled student* of ViT-g, whereas the ViT-L and
   larger encoders are natively pretrained under the JEPA objective. This is
   recorded in ``describe()`` so that the asymmetry is visible in the cache
   manifest rather than remembered.
"""

from __future__ import annotations

from typing import Any

import torch

from models.encoders import register_encoder
from models.encoders.base_encoder import BaseEncoder, EncoderOutput, PreprocessSpec, TokenLayout
from models.encoders.hf_common import (
    cfg_get,
    require_transformers,
    spatial_grid,
    temporal_grid,
    video_spec,
    vit_dims,
)


class VJEPA2Encoder(BaseEncoder):
    modality = "video"

    CHECKPOINTS = {
        "large": "facebook/vjepa2-vitl-fpc64-256",
        "huge": "facebook/vjepa2-vith-fpc64-256",
        "giant": "facebook/vjepa2-vitg-fpc64-256",
    }

    #: Variants produced by distillation rather than native JEPA pretraining.
    DISTILLED = {"base"}

    def __init__(
        self,
        variant: str = "large",
        *,
        model_name: str | None = None,
        model=None,
        frame_stride: int = 4,
        random_init: bool = False,
        freeze: bool = True,
    ) -> None:
        super().__init__(freeze=freeze)

        if model is None and random_init:
            require_transformers()
            from transformers import VJEPA2Config, VJEPA2Model

            dims = vit_dims(variant)
            model = VJEPA2Model(
                VJEPA2Config(
                    crop_size=224, patch_size=16, frames_per_clip=16, tubelet_size=2,
                    hidden_size=dims["hidden_size"],
                    num_hidden_layers=dims["num_hidden_layers"],
                    num_attention_heads=dims["num_attention_heads"],
                    mlp_ratio=4,
                    pred_hidden_size=384, pred_num_hidden_layers=12,
                    pred_num_attention_heads=12,
                )
            )
            model_name = f"random-init-{variant}"

        if model is None:
            require_transformers()
            from transformers import VJEPA2Model

            if model_name is None:
                if variant not in self.CHECKPOINTS:
                    raise ValueError(
                        f"Unknown variant {variant!r}. Available: "
                        f"{sorted(self.CHECKPOINTS)}. The distilled ViT-B is not "
                        f"listed because its identifier must be supplied "
                        f"explicitly via model_name; see docs/implementation_plan.md §10."
                    )
                model_name = self.CHECKPOINTS[variant]
            model = VJEPA2Model.from_pretrained(model_name)

        self.model = model
        self._model_name = model_name or "from-config"
        self._variant = variant

        cfg = model.config
        image_size = int(cfg_get(cfg, "crop_size", "image_size"))
        patch_size = int(cfg_get(cfg, "patch_size"))
        num_frames = int(cfg_get(cfg, "frames_per_clip", "num_frames"))
        tubelet = int(cfg_get(cfg, "tubelet_size"))

        h, w = spatial_grid(image_size, patch_size)
        t = temporal_grid(num_frames, tubelet)

        self._spec = video_spec(image_size, num_frames, frame_stride)
        self._layout = TokenLayout(
            grid=(t, h, w), dim=int(cfg_get(cfg, "hidden_size")), num_prefix_tokens=0
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
    def is_distilled(self) -> bool:
        return self._variant in self.DISTILLED

    def describe(self) -> dict[str, Any]:
        record = super().describe()
        record["variant"] = self._variant
        record["distilled"] = self.is_distilled
        return record

    def _forward_tokens(self, x: torch.Tensor) -> EncoderOutput:
        out = self.model(pixel_values_videos=x, skip_predictor=True)
        return EncoderOutput(tokens=out.last_hidden_state, prefix=None)


@register_encoder("vjepa2_l")
def _vjepa2_l(**kwargs) -> VJEPA2Encoder:
    return VJEPA2Encoder(variant="large", **kwargs)


@register_encoder("vjepa2_b")
def _vjepa2_b(model_name: str = "", **kwargs) -> VJEPA2Encoder:
    """Distilled ViT-B. The checkpoint identifier must be given explicitly."""
    return VJEPA2Encoder(variant="base", model_name=model_name, **kwargs)


__all__ = ["VJEPA2Encoder"]