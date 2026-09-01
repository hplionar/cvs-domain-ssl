"""Supervised ViT-B/16, as the control the objective comparison lacks.

Every other encoder in this registry is self-supervised, so the ranking they
produce cannot say whether self-supervision contributes anything over a strong
supervised backbone. That question is not idle: SwinCVS, the strongest
reproducible published result on Endoscapes, is supervised ImageNet transfer.

``google/vit-base-patch16-224`` is the canonical supervised ViT-B/16 --
pretrained on ImageNet-21k, fine-tuned on ImageNet-1k, 86M parameters, patch 16,
224 px, 196 patch tokens and a single CLS token. It matches dinov3_b and mae_b
exactly on architecture, patch size, token count, hidden dimension and input
resolution, which is the cell the objective comparison needs.

**What this arm can and cannot answer.** Pretraining corpus cannot be matched
alongside the objective: there is no supervised model trained on LVD-1689M,
because that corpus has no labels. The arm therefore answers whether a strong
publicly available supervised backbone is competitive with these self-supervised
encoders on CVS assessment, and not whether self-supervision helps. Report it as
the former. Its corpus does at least sit between the two self-supervised arms --
ImageNet-21k against DINOv3's LVD-1689M and ViT-MAE's ImageNet-1k -- so it is
not disadvantaged relative to both.

**Why not DeiT III**, which reaches roughly two points higher on ImageNet-1k:
it is distributed through ``timm`` rather than the Hub, which means a second
loading path, a second preprocessing convention and a new dependency, for a
baseline whose role is to establish whether supervised pretraining is
competitive at all. If this arm comes close to the self-supervised ones, the
comparison starts to hinge on recipe quality and DeiT III becomes worth the
integration cost. Not before.

**Preprocessing differs from the self-supervised arms and must not be
normalised away.** This checkpoint expects mean and standard deviation of 0.5 on
every channel, where DINOv2, DINOv3 and ViT-MAE expect the ImageNet statistics.
Those values are passed explicitly below rather than taken from ``image_spec``'s
ImageNet defaults, are recorded in the cache manifest, and ``verify_protocol``
will refuse to compare caches built under different transforms. That refusal is
correct: two encoders trained under different normalisation are not made
comparable by forcing one of them to see the wrong input distribution.

The classification head is discarded. ``ViTModel`` loads the encoder alone and
reports the two classifier tensors as unexpected, which is intended rather than
a defect.
"""

from __future__ import annotations

from typing import Any

from models.encoders import register_encoder
from models.encoders.base_encoder import BaseEncoder, PreprocessSpec, TokenLayout
from models.encoders.hf_common import (
    cfg_get,
    image_spec,
    require_transformers,
    spatial_grid,
)

#: This checkpoint family normalises to [-1, 1] rather than to the ImageNet
#: statistics, which is what ``image_spec`` defaults to.
VIT_MEAN = (0.5, 0.5, 0.5)
VIT_STD = (0.5, 0.5, 0.5)


class SupervisedViTEncoder(BaseEncoder):
    """Frozen supervised ViT, returning patch tokens and the CLS prefix."""

    modality = "image"

    CHECKPOINTS = {
        # ImageNet-21k pretraining, ImageNet-1k fine-tuning. The variant
        # everyone means by "supervised ViT-B/16".
        "base": "google/vit-base-patch16-224",
        "base_in21k": "google/vit-base-patch16-224-in21k",
        "large": "google/vit-large-patch16-224",
    }

    def __init__(
        self,
        variant: str = "base",
        model_name: str | None = None,
        model=None,
        random_init: bool = False,
        freeze: bool = True,
    ) -> None:
        super().__init__(freeze=freeze)

        if model is None:
            require_transformers()
            from transformers import ViTConfig, ViTModel

            if model_name is None:
                if variant not in self.CHECKPOINTS:
                    raise ValueError(
                        f"Unknown variant {variant!r}. Available: "
                        f"{sorted(self.CHECKPOINTS)}"
                    )
                model_name = self.CHECKPOINTS[variant]
            if random_init:
                # A control that isolates architecture from pretraining.
                model = ViTModel(ViTConfig.from_pretrained(model_name),
                                 add_pooling_layer=False)
            else:
                # add_pooling_layer=False: the pooler is a tanh projection of the
                # CLS token trained for classification, and is not part of the
                # representation under evaluation.
                model = ViTModel.from_pretrained(model_name, add_pooling_layer=False)

        self.model = model
        self._model_name = model_name or "unknown"
        self._variant = variant
        self._random_init = random_init

        config = model.config
        image_size = cfg_get(config, "image_size", default=224, required=False)
        patch_size = cfg_get(config, "patch_size", default=16, required=False)
        hidden = cfg_get(config, "hidden_size", default=768, required=False)

        self._layout = TokenLayout(
            grid=spatial_grid(image_size, patch_size),
            dim=hidden,
            # One CLS token, no registers and no distillation token: verified
            # against the checkpoint, which returns 197 tokens for a 14x14 grid.
            num_prefix_tokens=1,
        )
        self._spec = image_spec(image_size, mean=VIT_MEAN, std=VIT_STD)

        self._finalise_init()
        if freeze:
            self.freeze()

    @property
    def checkpoint_id(self) -> str:
        if self._random_init:
            return f"{self._model_name}+random_init"
        return self._model_name

    @property
    def token_layout(self) -> TokenLayout:
        return self._layout

    @property
    def preprocess_spec(self) -> PreprocessSpec:
        return self._spec

    def describe(self) -> dict[str, Any]:
        record = super().describe()
        record["supervision"] = "supervised classification"
        return record

    def _forward_tokens(self, pixel_values):
        return self.model(pixel_values=pixel_values).last_hidden_state


@register_encoder("vit_sup_b")
def _build_base(**kwargs) -> SupervisedViTEncoder:
    return SupervisedViTEncoder(variant="base", **kwargs)


@register_encoder("vit_sup_l")
def _build_large(**kwargs) -> SupervisedViTEncoder:
    return SupervisedViTEncoder(variant="large", **kwargs)
