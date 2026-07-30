"""VideoMAE encoder (masked pixel reconstruction, video level).

Represents the *masked reconstruction* arm of the E2 comparison. VideoMAE emits
tubelet tokens with no CLS token, so ``prefix`` is always ``None``.
"""

from __future__ import annotations

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


class VideoMAEEncoder(BaseEncoder):
    modality = "video"

    #: SSL-pretrained checkpoints. Note these are the *pretrain* variants, not
    #: the ``-finetuned-kinetics`` ones: supervised fine-tuning would contaminate
    #: the "original checkpoint" baseline that adaptation gain is measured from.
    CHECKPOINTS = {
        "base": "MCG-NJU/videomae-base",
        "large": "MCG-NJU/videomae-large",
    }

    def __init__(
        self,
        variant: str = "base",
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
            from transformers import VideoMAEConfig, VideoMAEModel

            model = VideoMAEModel(
                VideoMAEConfig(
                    image_size=224, patch_size=16, num_frames=16, tubelet_size=2,
                    **vit_dims(variant)
                )
            )
            model_name = f"random-init-{variant}"

        if model is None:
            require_transformers()
            from transformers import VideoMAEModel

            if model_name is None:
                if variant not in self.CHECKPOINTS:
                    raise ValueError(
                        f"Unknown variant {variant!r}. Available: {sorted(self.CHECKPOINTS)}"
                    )
                model_name = self.CHECKPOINTS[variant]
            model = VideoMAEModel.from_pretrained(model_name)

        self.model = model
        self._model_name = model_name or "from-config"
        self._frame_stride = frame_stride

        cfg = model.config
        image_size = int(cfg_get(cfg, "image_size"))
        patch_size = int(cfg_get(cfg, "patch_size"))
        num_frames = int(cfg_get(cfg, "num_frames"))
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

    def _forward_tokens(self, x: torch.Tensor) -> EncoderOutput:
        out = self.model(pixel_values=x)
        return EncoderOutput(tokens=out.last_hidden_state, prefix=None)


@register_encoder("videomae_b")
def _videomae_b(**kwargs) -> VideoMAEEncoder:
    return VideoMAEEncoder(variant="base", **kwargs)


@register_encoder("videomae_l")
def _videomae_l(**kwargs) -> VideoMAEEncoder:
    return VideoMAEEncoder(variant="large", **kwargs)


__all__ = ["VideoMAEEncoder"]