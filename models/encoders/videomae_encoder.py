"""VideoMAE encoder (masked pixel reconstruction, video level).

Represents the *masked reconstruction* arm of the E2 comparison. VideoMAE emits
tubelet tokens with no CLS token, so ``prefix`` is always ``None``.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path

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


def repair_qkv_bias(model, model_name: str) -> dict:
    """Restore attention biases dropped by the transformers 5.x refactor.

    VideoMAE follows the BEiT convention and stores attention bias as two
    separate parameters, ``q_bias`` and ``v_bias``, with the key bias fixed at
    zero:

        qkv_bias = cat([q_bias, zeros_like(v_bias), v_bias])

    ``VideoMAESelfAttention`` in transformers 5.x uses standard
    ``nn.Linear(bias=config.qkv_bias)``, expecting ``query.bias``, ``key.bias``
    and ``value.bias``. No conversion exists between the two layouts, so
    ``from_pretrained`` reports the checkpoint's ``q_bias``/``v_bias`` as
    UNEXPECTED and the Linear biases as MISSING, silently substituting freshly
    initialised values for trained ones.

    This matters here because adaptation gain is measured relative to the
    original checkpoint. An encoder loaded with the wrong biases is not the
    published encoder, and both the baseline and the adapted run would be
    affected in unknown directions.

    Returns a record of what was repaired, which is written into the cache
    manifest so the state is visible rather than assumed.
    """
    import re

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:  # pragma: no cover
        return {"status": "skipped", "reason": "huggingface_hub unavailable"}

    raw = None
    for filename in ("model.safetensors", "pytorch_model.bin"):
        try:
            path = hf_hub_download(model_name, filename)
        except Exception:  # noqa: BLE001 - try the next candidate
            continue
        if filename.endswith(".safetensors"):
            from safetensors.torch import load_file

            raw = load_file(path)
        else:
            raw = torch.load(path, map_location="cpu", weights_only=True)
        break

    if raw is None:
        return {"status": "skipped", "reason": f"no checkpoint file for {model_name}"}

    # Both the checkpoint and the model may or may not carry a `videomae.`
    # prefix, depending on whether they came from VideoMAEModel (encoder only)
    # or VideoMAEForPreTraining (encoder plus decoder). Stripping the prefix
    # unconditionally works for the former and silently fails for the latter,
    # so candidates are tried both ways against the model's own keys.
    sources = {
        key: value for key, value in raw.items() if key.endswith(("q_bias", "v_bias"))
    }
    if not sources:
        return {"status": "not_needed", "reason": "checkpoint has no q_bias/v_bias"}

    state = model.state_dict()
    repaired, missing = [], []

    for key, value in sources.items():
        base = key.replace(".q_bias", ".query.bias").replace(".v_bias", ".value.bias")
        stripped = re.sub(r"^videomae\.", "", base)
        target = next(
            (c for c in (base, stripped, f"videomae.{stripped}") if c in state), None
        )
        if target is None:
            missing.append(base)
            continue
        if state[target].shape != value.shape:
            missing.append(f"{target} (shape {tuple(state[target].shape)} vs {tuple(value.shape)})")
            continue
        state[target] = value.to(state[target].dtype)
        repaired.append(target)

        # The original implementation fixes the key bias at zero. Setting it
        # explicitly rather than leaving it at whatever init produced.
        key_bias = target.replace(".query.bias", ".key.bias").replace(".value.bias", ".key.bias")
        if key_bias in state:
            state[key_bias] = torch.zeros_like(state[key_bias])

    if missing:
        warnings.warn(
            f"Could not map {len(missing)} attention bias tensors for {model_name}: "
            f"{missing[:4]}. The encoder may not match the published checkpoint.",
            RuntimeWarning,
            stacklevel=2,
        )

    model.load_state_dict(state, strict=True)

    return {
        "status": "repaired" if repaired else "failed",
        "num_repaired": len(repaired),
        "num_unmapped": len(missing),
        "reason": (
            "transformers 5.x expects query/key/value.bias; VideoMAE checkpoints "
            "store q_bias/v_bias with key bias fixed at zero"
        ),
    }


def load_adapted_checkpoint(
    path: str | Path,
    *,
    base_checkpoint: str | None = None,
    fallback_base: str | None = None,
):
    """Rebuild a ``VideoMAEModel`` carrying weights from continued pretraining.

    ``train/pretrain_videomae.py`` exports ``encoder_final.pt`` as a bare state
    dict of ``VideoMAEForPreTraining.videomae``, wrapped alongside the training
    config. That is not a HuggingFace directory, so ``from_pretrained`` cannot
    read it and the adapted arm has no route into ``extract_features.py``.

    The architecture is reconstructed from the *base* checkpoint, whose
    identifier the training config already records, and the adapted weights are
    then loaded over it.

    Bias handling
    -------------
    ``repair_qkv_bias`` is deliberately **not** applied here. Pretraining
    repaired the biases before training began, so the adapted state dict already
    carries trained ``query.bias``/``value.bias`` in the transformers 5.x
    layout. Re-running the repair would overwrite trained values with the
    original checkpoint's, silently undoing part of the adaptation.

    Loading is non-strict so that missing and unexpected keys can be reported
    separately. Missing encoder keys are fatal: they mean the probe would run on
    freshly initialised weights, which is the failure this whole path exists to
    prevent.

    Returns
    -------
    ``(model, base_checkpoint, record)`` where ``record`` is written into the
    cache manifest.
    """
    require_transformers()
    from transformers import VideoMAEModel

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No adapted checkpoint at {path}.")

    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:  # noqa: BLE001 - older checkpoints may carry non-tensor config
        payload = torch.load(path, map_location="cpu", weights_only=False)

    if not isinstance(payload, dict) or "model" not in payload:
        raise ValueError(
            f"{path} does not look like a checkpoint written by "
            f"pretrain_videomae.py: expected a dict with a 'model' key, got "
            f"{type(payload).__name__}"
            + (f" with keys {sorted(payload)[:8]}" if isinstance(payload, dict) else "")
            + "."
        )

    state = dict(payload["model"])
    # A resume checkpoint (`latest.pt`) holds VideoMAEForPreTraining, whose keys
    # carry a `videomae.` prefix and include decoder weights. `encoder_final.pt`
    # holds the encoder alone. Accept either.
    if any(key.startswith("videomae.") for key in state):
        state = {
            re.sub(r"^videomae\.", "", key): value
            for key, value in state.items()
            if not key.startswith("decoder.")
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

    model = VideoMAEModel.from_pretrained(base)
    result = model.load_state_dict(state, strict=False)

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
            f"VideoMAEModel and were ignored, e.g. {result.unexpected_keys[:5]}.",
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
        # Written by save_checkpoint for resume checkpoints; absent from
        # encoder_final.pt, which is a plain torch.save.
        "bias_repair_at_pretrain": payload.get("bias_repair"),
    }
    return model, base, record


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
        adapted_checkpoint: str | Path | None = None,
        model=None,
        frame_stride: int = 4,
        random_init: bool = False,
        freeze: bool = True,
    ) -> None:
        super().__init__(freeze=freeze)

        self._adaptation: dict | None = None

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
            # Distinguishable from the baseline arm in the cache manifest, which
            # is the whole point of separating these two measurements.
            model_name = f"{base}+adapted:{Path(adapted_checkpoint).name}"
            self._bias_repair = {
                "status": "inherited_from_adapted_checkpoint",
                "reason": (
                    "biases were repaired before pretraining and are carried by "
                    "the adapted state dict; re-repairing would restore the base "
                    "checkpoint's untrained values"
                ),
            }

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
            self._bias_repair = repair_qkv_bias(model, model_name)
        elif not hasattr(self, "_bias_repair"):
            # A model object was supplied directly; the adapted path has already
            # recorded its own state.
            self._bias_repair = {"status": "not_attempted"}

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

    @property
    def is_adapted(self) -> bool:
        return self._adaptation is not None

    def describe(self) -> dict:
        record = super().describe()
        record["qkv_bias_repair"] = self._bias_repair
        record["adaptation"] = self._adaptation
        return record

    def _forward_tokens(
        self,
        x: torch.Tensor,
        *,
        layer_depths: tuple[float, ...] | None = None,
    ) -> EncoderOutput:
        if layer_depths is None:
            out = self.model(pixel_values=x)
            return EncoderOutput(tokens=out.last_hidden_state, prefix=None)

        from models.encoders.base_encoder import resolve_layer_indices

        out = self.model(pixel_values=x, output_hidden_states=True)
        # VideoMAE emits no prefix token, so every hidden state is already pure
        # patch tokens and needs no slicing. Encoders with a CLS or register
        # tokens must strip them here.
        indices = resolve_layer_indices(
            layer_depths, self.model.config.num_hidden_layers
        )
        selected = torch.stack([out.hidden_states[i] for i in indices], dim=1)
        return EncoderOutput(
            tokens=out.last_hidden_state,
            prefix=None,
            hidden_states=selected,
        )


@register_encoder("videomae_b")
def _videomae_b(**kwargs) -> VideoMAEEncoder:
    return VideoMAEEncoder(variant="base", **kwargs)


@register_encoder("videomae_l")
def _videomae_l(**kwargs) -> VideoMAEEncoder:
    return VideoMAEEncoder(variant="large", **kwargs)


__all__ = ["VideoMAEEncoder"]