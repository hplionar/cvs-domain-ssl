"""VideoMAE encoder (masked pixel reconstruction, video level).

Represents the *masked reconstruction* arm of the E2 comparison. VideoMAE emits
tubelet tokens with no CLS token, so ``prefix`` is always ``None``.
"""

from __future__ import annotations

import warnings

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
            self._bias_repair = repair_qkv_bias(model, model_name)
        else:
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

    def describe(self) -> dict:
        record = super().describe()
        record["qkv_bias_repair"] = self._bias_repair
        return record

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