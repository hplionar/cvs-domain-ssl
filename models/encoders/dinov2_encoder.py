"""DINOv2 image encoders. Two of them, for different pipelines.

``DINOv2Encoder`` (legacy)
    torch.hub wrapper returning a pooled ``[B, D]`` vector. It is *not* a
    ``BaseEncoder`` and is not in the registry. ``train/train_cvs.py``,
    ``train/train_cvs_clip_meanpool.py`` and both scripts in ``eval/`` import it
    directly, and exp001-exp006 were produced through it, so it is kept exactly
    as it was. Pooling inside the encoder is irreversible, so it cannot feed any
    head in ``models/heads/``.

``DINOv2ViTEncoder``
    ``BaseEncoder`` returning a ``[B, N, D]`` token grid with the ``[CLS]`` token
    separated into ``prefix``. This is the one the cached-probe path uses.
    Registered as ``dinov2_s``, ``dinov2_b``, ``dinov2_l``.

The weights are **not gated**: ``facebook/dinov2-base`` downloads without a
token or an access request, unlike the DINOv3 family.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from typing import Literal

import torch
import torch.nn as nn

from models.encoders import register_encoder
from models.encoders.base_encoder import BaseEncoder, EncoderOutput, PreprocessSpec, TokenLayout
from models.encoders.hf_common import (
    cfg_get,
    image_spec,
    require_transformers,
    spatial_grid,
    vit_dims,
)


DINO_MODEL_NAMES = {
    "small": "dinov2_vits14",
    "base": "dinov2_vitb14",
    "large": "dinov2_vitl14",
    "giant": "dinov2_vitg14",
}


class DINOv2Encoder(nn.Module):
    """
    DINOv2 image encoder wrapper.

    Input:
        images: Tensor of shape [B, 3, H, W]

    Output:
        features: Tensor of shape [B, feature_dim]

    Notes:
        This wrapper uses torch.hub to load the official DINOv2 models.
        For the first baseline, use variant='base'.
    """

    def __init__(
        self,
        variant: Literal["small", "base", "large", "giant"] = "base",
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        if variant not in DINO_MODEL_NAMES:
            raise ValueError(
                f"Unknown DINOv2 variant: {variant}. "
                f"Available variants: {list(DINO_MODEL_NAMES.keys())}"
            )

        self.variant = variant
        self.model_name = DINO_MODEL_NAMES[variant]

        self.encoder = torch.hub.load(
            "facebookresearch/dinov2",
            self.model_name,
            pretrained=pretrained,
        )

        self.feature_dim = self._infer_feature_dim()

    def _infer_feature_dim(self) -> int:
        feature_dims = {
            "small": 384,
            "base": 768,
            "large": 1024,
            "giant": 1536,
        }
        return feature_dims[self.variant]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.encoder(x)

        if not isinstance(output, torch.Tensor):
            raise TypeError(f"Expected DINOv2 output to be a Tensor, got {type(output)}")

        if output.ndim != 2:
            raise ValueError(
                f"Expected DINOv2 features with shape [B, D], got {tuple(output.shape)}"
            )

        return output

# ---------------------------------------------------------------------------
# BaseEncoder wrapper: token grids for the cached-probe path
# ---------------------------------------------------------------------------


def load_adapted_checkpoint(
    path: str | Path,
    *,
    base_checkpoint: str | None = None,
    fallback_base: str | None = None,
    model_cls=None,
):
    """Rebuild a ``Dinov2Model`` carrying weights from continued pretraining.

    ``train/pretrain_dino.py`` exports ``encoder_final.pt`` as a bare state dict
    of the **teacher** backbone, wrapped alongside the training config. That is
    not a HuggingFace directory, so ``from_pretrained`` cannot read it and the
    adapted arm has no route into ``extract_features.py``.

    The teacher rather than the student is exported deliberately: it is the EMA
    of the student, it is what DINO's own linear-probe protocol evaluates, and
    collapse is a property of the teacher.

    Unlike the VideoMAE counterpart there is no bias repair, because DINOv2
    stores query, key and value biases under the names transformers expects.
    Unlike the V-JEPA counterpart the exported weights are the whole backbone
    rather than a submodule, so no prefix needs stripping from
    ``encoder_final.pt`` -- though ``latest.pt`` does carry one.

    Returns ``(model, base_checkpoint, record)``, the record going into the
    cache manifest so that an adapted cache is distinguishable from a baseline
    one after the fact.
    """
    require_transformers()
    if model_cls is None:
        from transformers import Dinov2Model

        model_cls = Dinov2Model

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No adapted checkpoint at {path}.")

    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:  # noqa: BLE001 - checkpoints carry non-tensor config
        payload = torch.load(path, map_location="cpu", weights_only=False)

    if not isinstance(payload, dict):
        raise ValueError(
            f"{path} does not look like a checkpoint written by "
            f"pretrain_dino.py: expected a dict, got {type(payload).__name__}."
        )

    # encoder_final.pt carries the teacher backbone under "model".
    # latest.pt carries the whole DINOModel under "teacher", whose keys are
    # prefixed "backbone." and "head.".
    if "model" in payload:
        state = dict(payload["model"])
        source = "encoder_final"
    elif "teacher" in payload:
        state = dict(payload["teacher"])
        source = "latest"
    else:
        raise ValueError(
            f"{path} has neither a 'model' nor a 'teacher' key; found "
            f"{sorted(payload)[:8]}. It was not written by pretrain_dino.py."
        )

    if any(key.startswith("backbone.") for key in state):
        # Keep the backbone, drop the projection head: the head is a
        # pretraining artefact and extraction never runs it.
        state = {
            key[len("backbone.") :]: value
            for key, value in state.items()
            if key.startswith("backbone.")
        }

    recorded_base = (payload.get("config") or {}).get("model", {}).get("checkpoint")
    base = base_checkpoint or recorded_base or fallback_base
    if base is None:
        raise ValueError(
            f"{path} records no base checkpoint under config.model.checkpoint, "
            f"and none was supplied. Pass model_name explicitly so the "
            f"architecture can be reconstructed."
        )
    if base_checkpoint and recorded_base and base_checkpoint != recorded_base:
        raise ValueError(
            f"{path} was trained from {recorded_base!r} but {base_checkpoint!r} "
            f"was requested. Loading adapted weights into a different "
            f"architecture would either fail loudly or, worse, partially "
            f"succeed."
        )

    model = model_cls.from_pretrained(base)
    result = model.load_state_dict(state, strict=False)

    missing = list(result.missing_keys)
    unexpected = list(result.unexpected_keys)
    if missing:
        raise ValueError(
            f"{len(missing)} parameters in the {base} architecture were not "
            f"present in {path.name}: {missing[:5]}. The adapted encoder would "
            f"carry a mixture of trained and pretrained weights, which is not a "
            f"state any arm of the comparison is supposed to be in."
        )

    record = {
        "path": str(path),
        "base_checkpoint": base,
        "base_recorded_in_checkpoint": recorded_base,
        "source_key": source,
        "step": payload.get("step"),
        "num_loaded": len(state),
        "num_ignored": len(unexpected),
        # The teacher is the EMA of the student. Recorded because a reader of
        # the manifest cannot otherwise tell which of the two was evaluated.
        "exported": "teacher_backbone",
    }
    return model, base, record


class DINOv2ViTEncoder(BaseEncoder):
    """DINOv2 ViT returning patch tokens with ``[CLS]`` separated.

    Self-distillation, image level, and **ungated** -- which is the practical
    reason it exists alongside ``dinov3_*``: the DINOv3 checkpoints sit behind a
    manual access review, and a frozen-feature head comparison does not need to
    wait for one. Both families are self-distilled, so both give a ``[CLS]``
    token that carries a real global summary rather than the near-empty one a
    reconstruction objective produces.

    Resolution
    ----------
    ``image_size`` is a constructor argument here, not read from the checkpoint
    config as it is for DINOv3. The released config declares 518, which at
    patch 14 is a 37x37 grid: 1369 tokens per image, seven times the cache and
    the compute, at a resolution nothing else in this project uses. DINOv2
    interpolates its position encodings for any input size, so 224 is a valid
    input and yields a 16x16 grid.

    That interpolation is silent, so the guard below is the only thing standing
    between a mistyped resolution and a cache built at the wrong geometry.
    ``image_size`` must be divisible by the patch size, which for the /14
    checkpoints excludes some otherwise natural choices -- 256 is not a multiple
    of 14, 224 and 252 are.

    Register tokens
    ---------------
    The original DINOv2 release has none, so ``prefix`` is a single ``[CLS]``.
    ``facebook/dinov2-with-registers-*`` is a separate checkpoint family that has
    four, and loading one under an assumption of zero would shift every patch
    token by four positions without raising anything. So the count is not
    assumed -- but neither is it read from the config, because the config lies:

        >>> cfg = Dinov2Config(num_register_tokens=4)   # accepted
        >>> Dinov2Model(cfg)(pixel_values=x).last_hidden_state.shape[1]
        257                                             # 256 patches + 1 CLS

    ``Dinov2Config`` stores the field, ``Dinov2Model`` has no ``register_tokens``
    parameter and ignores it. Only ``Dinov2WithRegistersModel`` implements them.
    The count is therefore taken from the *architecture* -- whether the
    embedding module actually owns register tokens -- and the config is
    consulted only once that is established. ``AutoModel`` picks the right class
    per checkpoint, so both families load correctly by name.

    The shape check in ``_forward_tokens`` is the backstop for all of this, and
    is what caught the config-lies case in the first place.
    """

    modality = "image"

    CHECKPOINTS = {
        "small": "facebook/dinov2-small",
        "base": "facebook/dinov2-base",
        "large": "facebook/dinov2-large",
        "giant": "facebook/dinov2-giant",
    }

    #: Resolution used across this project. Not the checkpoint's declared 518.
    DEFAULT_IMAGE_SIZE = 224

    def __init__(
        self,
        variant: str = "base",
        *,
        image_size: int | None = None,
        model_name: str | None = None,
        model=None,
        random_init: bool = False,
        adapted_checkpoint: str | Path | None = None,
        freeze: bool = True,
    ) -> None:
        super().__init__(freeze=freeze)

        self._adaptation: dict[str, Any] | None = None
        if adapted_checkpoint is not None:
            if model is not None or random_init:
                raise ValueError(
                    "adapted_checkpoint cannot be combined with model= or "
                    "random_init=True."
                )
            model, base, self._adaptation = load_adapted_checkpoint(
                adapted_checkpoint,
                base_checkpoint=model_name,
                fallback_base=self.CHECKPOINTS.get(variant),
            )
            model_name = f"{base}+adapted:{Path(adapted_checkpoint).name}"

        if model is None and random_init:
            require_transformers()
            from transformers import Dinov2Config, Dinov2Model

            model = Dinov2Model(
                Dinov2Config(image_size=518, patch_size=14, **vit_dims(variant))
            )
            model_name = f"random-init-{variant}"

        if model is None:
            require_transformers()
            from transformers import AutoModel

            if model_name is None:
                if variant not in self.CHECKPOINTS:
                    raise ValueError(
                        f"Unknown variant {variant!r}. Available: {sorted(self.CHECKPOINTS)}"
                    )
                model_name = self.CHECKPOINTS[variant]
            # AutoModel rather than Dinov2Model: the with-registers checkpoints
            # need Dinov2WithRegistersModel, and loading them through
            # Dinov2Model would drop the register weights silently.
            model = AutoModel.from_pretrained(model_name)

        self.model = model
        self._model_name = model_name or "from-config"

        cfg = model.config
        patch_size = int(cfg_get(cfg, "patch_size"))
        size = int(self.DEFAULT_IMAGE_SIZE if image_size is None else image_size)
        if size % patch_size != 0:
            raise ValueError(
                f"image_size {size} is not divisible by DINOv2's patch size "
                f"{patch_size}. The model would interpolate its position "
                f"encodings and return a token count that does not match the "
                f"declared grid. Use a multiple of {patch_size}: "
                f"{patch_size * (size // patch_size)} or "
                f"{patch_size * (size // patch_size + 1)}."
            )
        h, w = spatial_grid(size, patch_size)

        # Architecture first, config second. Dinov2Config accepts and stores
        # num_register_tokens that Dinov2Model does not implement, so trusting
        # the config alone declares a prefix the model never emits and every
        # patch token lands four positions out.
        embeddings = getattr(model, "embeddings", None)
        has_registers = getattr(embeddings, "register_tokens", None) is not None
        num_registers = (
            int(cfg_get(cfg, "num_register_tokens", default=0, required=False) or 0)
            if has_registers
            else 0
        )
        self._num_registers = num_registers

        self._spec = image_spec(size)
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

    def describe(self) -> dict:
        """Recorded in the cache manifest.

        Without the adaptation record, a cache built from an adapted checkpoint
        is indistinguishable from a baseline one after the fact, and the two
        arms of a comparison cannot be told apart from their manifests alone.
        """
        record = super().describe()
        if getattr(self, "_adaptation", None) is not None:
            record["adaptation"] = self._adaptation
        return record

    @property
    def checkpoint_id(self) -> str:
        return f"{self._model_name}@{self._spec.image_size}"

    @property
    def num_register_tokens(self) -> int:
        return self._num_registers

    def _forward_tokens(self, x: torch.Tensor) -> EncoderOutput:
        hidden = self.model(pixel_values=x).last_hidden_state
        n_prefix = self._layout.num_prefix_tokens
        expected = self._layout.num_tokens + n_prefix
        if hidden.shape[1] != expected:
            raise RuntimeError(
                f"DINOv2 returned {hidden.shape[1]} tokens, expected {expected} "
                f"({n_prefix} prefix + {self._layout.num_tokens} patch). Position "
                f"encodings are interpolated silently, so this means the input "
                f"resolution or the register count disagrees with the declared "
                f"layout; patch tokens would be misaligned."
            )
        return EncoderOutput(tokens=hidden[:, n_prefix:, :], prefix=hidden[:, :n_prefix, :])


@register_encoder("dinov2_s")
def _dinov2_s(**kwargs) -> DINOv2ViTEncoder:
    return DINOv2ViTEncoder(variant="small", **kwargs)


@register_encoder("dinov2_b")
def _dinov2_b(**kwargs) -> DINOv2ViTEncoder:
    return DINOv2ViTEncoder(variant="base", **kwargs)


@register_encoder("dinov2_l")
def _dinov2_l(**kwargs) -> DINOv2ViTEncoder:
    return DINOv2ViTEncoder(variant="large", **kwargs)


__all__ = ["DINO_MODEL_NAMES", "DINOv2Encoder", "DINOv2ViTEncoder"]
