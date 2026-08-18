"""I-JEPA encoder (latent prediction, image level).

The image-level counterpart to V-JEPA 2, and the fourth cell of the objective
family by modality design:

    ================  ========================  ====================
                      masked reconstruction     latent prediction
    ================  ========================  ====================
    image             ViT-MAE                   I-JEPA
    video             VideoMAE                  V-JEPA 2
    ================  ========================  ====================

Capacity, and why this arm is not matched
-----------------------------------------
**The released checkpoints start at ViT-H.** Facebook publishes
``ijepa_vith14_1k``, ``ijepa_vith14_22k``, ``ijepa_vith16_1k`` and
``ijepa_vitg16_22k``; there is no ViT-B or ViT-L. At 632M parameters against
DINOv2 ViT-B's 86M, this arm is therefore **not capacity-matched** to the other
image encoders and any advantage it shows is confounded with size.

(``sparsh-ijepa-base`` and ``sparsh-ijepa-small`` are Meta's tactile-sensing
models, not vision encoders, and are not substitutes.)

This mirrors the situation already present in the video arms, where V-JEPA 2
ViT-L (326M) is compared against VideoMAE ViT-B (86M) because no smaller V-JEPA
2 release exists either. In both modalities the latent-prediction arm is the
larger model. That is a limitation of what has been published, not a design
choice, and it must be stated wherever these arms are compared.

Architecture notes
------------------
I-JEPA emits **no CLS token**: at 224 px with patch 14 the output is exactly
256 patch tokens, verified rather than assumed by the guard in
``_forward_tokens``. ``num_prefix_tokens`` is therefore 0, as for VideoMAE, and
any head whose global branch needs a CLS must fall back to mean-pooled patches.

Unlike ``ViTMAEModel``, ``IJepaModel`` performs no masking during the forward
pass, so there is no equivalent of the monotonic-noise workaround needed there.
The context encoder is what is published; the predictor used during pretraining
is not part of the released model.
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
)


class IJEPAEncoder(BaseEncoder):
    modality = "image"

    CHECKPOINTS = {
        # ImageNet-22k by default: it saw substantially more data than the 1k
        # variants and is the closer counterpart to DINOv2's large curated
        # corpus, which keeps the image arms comparable on pretraining scale
        # even though they cannot be matched on capacity.
        "huge": "facebook/ijepa_vith14_22k",
        "huge_1k": "facebook/ijepa_vith14_1k",
        "huge_p16": "facebook/ijepa_vith16_1k",
        "giant": "facebook/ijepa_vitg16_22k",
    }

    def __init__(
        self,
        variant: str = "huge",
        *,
        model_name: str | None = None,
        model=None,
        random_init: bool = False,
        freeze: bool = True,
    ) -> None:
        super().__init__(freeze=freeze)

        if model is None and random_init:
            require_transformers()
            from transformers import IJepaConfig, IJepaModel

            model = IJepaModel(
                IJepaConfig(
                    image_size=224,
                    patch_size=14,
                    hidden_size=1280,
                    num_hidden_layers=32,
                    num_attention_heads=16,
                    intermediate_size=5120,
                )
            )
            model_name = f"random-init-{variant}"

        if model is None:
            require_transformers()
            from transformers import IJepaModel

            if model_name is None:
                if variant not in self.CHECKPOINTS:
                    raise ValueError(
                        f"Unknown variant {variant!r}. Available: "
                        f"{sorted(self.CHECKPOINTS)}"
                    )
                model_name = self.CHECKPOINTS[variant]
            model = IJepaModel.from_pretrained(model_name)

        self.model = model
        self._model_name = model_name or "unknown"
        self._variant = variant

        config = model.config
        image_size = cfg_get(config, "image_size", 224)
        patch_size = cfg_get(config, "patch_size", 14)
        hidden = cfg_get(config, "hidden_size", 1280)

        if image_size % patch_size:
            raise ValueError(
                f"image_size {image_size} is not divisible by patch_size "
                f"{patch_size}; the token grid would not be rectangular."
            )

        self._spec = image_spec(image_size)
        self._layout = TokenLayout(
            grid=spatial_grid(image_size, patch_size),
            dim=hidden,
            # I-JEPA has no CLS token. Verified in _forward_tokens rather than
            # trusted, so a future architecture change surfaces as an error
            # instead of a silently misaligned cache.
            num_prefix_tokens=0,
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

    def describe(self) -> dict:
        record = super().describe()
        record["variant"] = self._variant
        record["capacity_note"] = (
            "I-JEPA releases start at ViT-H (632M). This arm is not "
            "capacity-matched to the ViT-B image encoders and any advantage is "
            "confounded with model size."
        )
        return record

    def _forward_tokens(
        self,
        x: torch.Tensor,
        *,
        layer_depths: tuple[float, ...] | None = None,
    ) -> EncoderOutput:
        if layer_depths is not None:
            raise NotImplementedError(
                "IJEPAEncoder does not yet support layer_depths. Returning the "
                "final layer silently would make a depth comparison "
                "meaningless."
            )

        out = self.model(pixel_values=x)
        hidden = out.last_hidden_state

        expected = self._layout.num_tokens
        if hidden.shape[1] != expected:
            raise RuntimeError(
                f"I-JEPA returned {hidden.shape[1]} tokens, expected {expected} "
                f"for a {self._layout.grid} grid. Either a prefix token has "
                f"appeared -- in which case num_prefix_tokens must be updated "
                f"and the leading tokens stripped -- or the input resolution "
                f"does not match the checkpoint. Caching this would misalign "
                f"every token with its spatial position."
            )

        return EncoderOutput(tokens=hidden, prefix=None)


@register_encoder("ijepa_h")
def _ijepa_h(**kwargs) -> IJEPAEncoder:
    return IJEPAEncoder(variant="huge", **kwargs)


@register_encoder("ijepa_h_1k")
def _ijepa_h_1k(**kwargs) -> IJEPAEncoder:
    return IJEPAEncoder(variant="huge_1k", **kwargs)


@register_encoder("ijepa_g")
def _ijepa_g(**kwargs) -> IJEPAEncoder:
    return IJEPAEncoder(variant="giant", **kwargs)
