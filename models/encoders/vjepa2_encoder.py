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

import warnings
from pathlib import Path
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


def load_adapted_checkpoint(
    path: str | Path,
    *,
    base_checkpoint: str | None = None,
    fallback_base: str | None = None,
):
    """Rebuild a ``VJEPA2Model`` carrying weights from continued pretraining.

    ``train/pretrain_vjepa2.py`` exports ``encoder_final.pt`` as a bare state
    dict of ``VJEPA2Model.encoder``, wrapped alongside the training config. That
    is not a HuggingFace directory, so ``from_pretrained`` cannot read it and the
    adapted arm has no route into ``extract_features.py``.

    The counterpart of ``videomae_encoder.load_adapted_checkpoint``, differing in
    two respects. There is no bias repair, because V-JEPA stores query, key and
    value biases under the names transformers expects. And the encoder is a
    submodule rather than the whole model, so weights load into ``model.encoder``
    while the predictor keeps its pretrained values — the predictor is the SSL
    head, not part of the representation, and feature extraction never runs it.

    Returns ``(model, base_checkpoint, record)`` where ``record`` goes into the
    cache manifest.
    """
    require_transformers()
    from transformers import VJEPA2Model

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No adapted checkpoint at {path}.")

    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:  # noqa: BLE001 - checkpoints may carry non-tensor config
        payload = torch.load(path, map_location="cpu", weights_only=False)

    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError(
            f"{path} does not look like a checkpoint written by "
            f"pretrain_vjepa2.py: expected a dict with a 'model' key, got "
            f"{type(payload).__name__}"
            + (f" with keys {sorted(payload)[:8]}" if isinstance(payload, dict) else "")
            + "."
        )

    state = dict(payload["model"])
    # `latest.pt` holds the whole VJEPA2Model, whose keys carry an `encoder.`
    # prefix and include the predictor. `encoder_final.pt` holds the encoder
    # alone. Accept either, keeping only encoder weights.
    if any(key.startswith("encoder.") for key in state):
        state = {
            key[len("encoder.") :]: value
            for key, value in state.items()
            if key.startswith("encoder.")
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
        warnings.warn(
            f"Base checkpoint {base_checkpoint!r} was supplied but {path} "
            f"records {recorded_base!r}. Using the supplied value; verify this "
            f"is intended, since a geometry mismatch will surface as missing "
            f"keys rather than a clear error.",
            RuntimeWarning,
            stacklevel=2,
        )

    model = VJEPA2Model.from_pretrained(base)
    result = model.encoder.load_state_dict(state, strict=False)

    if result.missing_keys:
        raise RuntimeError(
            f"{len(result.missing_keys)} encoder parameters were not present in "
            f"{path}, e.g. {result.missing_keys[:5]}. Those weights would remain "
            f"at the base checkpoint's values while the rest are adapted, which "
            f"is not a valid arm of the comparison."
        )
    if result.unexpected_keys:
        warnings.warn(
            f"{len(result.unexpected_keys)} keys in {path} had no counterpart in "
            f"VJEPA2Model.encoder and were ignored, e.g. {result.unexpected_keys[:5]}.",
            RuntimeWarning,
            stacklevel=2,
        )

    record = {
        "path": str(path),
        "base_checkpoint": base,
        "base_recorded_in_checkpoint": recorded_base,
        "step": payload.get("step"),
        "num_loaded": len(state) - len(result.unexpected_keys),
        "num_ignored": len(result.unexpected_keys),
        "predictor": "retained from base checkpoint; not used for extraction",
    }
    return model, base, record



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
        if self._adaptation is not None:
            record["adaptation"] = self._adaptation
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