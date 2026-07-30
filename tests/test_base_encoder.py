"""Contract tests for models.encoders.base_encoder.

These use synthetic encoders rather than real checkpoints so that the interface
can be validated without GPU access or network downloads.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from models.encoders.base_encoder import (
    BaseEncoder,
    EncoderOutput,
    PreprocessSpec,
    TokenLayout,
)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class DummyImageEncoder(BaseEncoder):
    """ViT-B/16 at 224 px: 14x14 patch grid, one CLS plus four register tokens."""

    modality = "image"

    def __init__(self, *, freeze: bool = True, num_prefix: int = 5) -> None:
        super().__init__(freeze=freeze)
        self.proj = nn.Linear(3 * 16 * 16, 768)
        self._num_prefix = num_prefix
        self.prefix_embed = nn.Parameter(torch.zeros(num_prefix, 768))
        self._finalise_init()

    @property
    def preprocess_spec(self) -> PreprocessSpec:
        return PreprocessSpec(image_size=224, mean=IMAGENET_MEAN, std=IMAGENET_STD)

    @property
    def token_layout(self) -> TokenLayout:
        return TokenLayout(grid=(14, 14), dim=768, num_prefix_tokens=self._num_prefix)

    def _forward_tokens(self, x: torch.Tensor) -> EncoderOutput:
        b = x.shape[0]
        patches = x.unfold(2, 16, 16).unfold(3, 16, 16)
        patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(b, 14 * 14, -1)
        tokens = self.proj(patches)
        prefix = self.prefix_embed.unsqueeze(0).expand(b, -1, -1)
        return EncoderOutput(tokens=tokens, prefix=prefix)


class DummyVideoEncoder(BaseEncoder):
    """16 frames, tubelet 2, patch 16 at 224 px: 8x14x14 grid, no prefix."""

    modality = "video"

    def __init__(self, *, freeze: bool = True) -> None:
        super().__init__(freeze=freeze)
        self.proj = nn.Linear(3 * 2 * 16 * 16, 768)
        self._finalise_init()

    @property
    def preprocess_spec(self) -> PreprocessSpec:
        return PreprocessSpec(
            image_size=224,
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
            num_frames=16,
            frame_stride=4,
        )

    @property
    def token_layout(self) -> TokenLayout:
        return TokenLayout(grid=(8, 14, 14), dim=768, num_prefix_tokens=0)

    def _forward_tokens(self, x: torch.Tensor) -> EncoderOutput:
        b = x.shape[0]
        n = 8 * 14 * 14
        tubelets = (
            x.reshape(b, 8, 2, 3, 224, 224)
            .unfold(4, 16, 16)
            .unfold(5, 16, 16)
            .permute(0, 1, 4, 5, 3, 2, 6, 7)
            .reshape(b, n, -1)
        )
        return EncoderOutput(tokens=self.proj(tubelets), prefix=None)


class LeakyPrefixEncoder(DummyImageEncoder):
    """Contract violation: returns prefix tokens concatenated into `tokens`."""

    def _forward_tokens(self, x: torch.Tensor) -> EncoderOutput:
        out = super()._forward_tokens(x)
        return EncoderOutput(tokens=torch.cat([out.prefix, out.tokens], dim=1), prefix=None)


# -- layout ---------------------------------------------------------------


def test_token_layout_counts_and_footprint():
    image = TokenLayout(grid=(14, 14), dim=768, num_prefix_tokens=5)
    assert image.num_tokens == 196
    assert not image.is_spatiotemporal
    assert image.bytes_per_sample() == 196 * 768 * 2

    video = TokenLayout(grid=(8, 14, 14), dim=768)
    assert video.num_tokens == 1568
    assert video.is_spatiotemporal


@pytest.mark.parametrize(
    "kwargs",
    [
        {"grid": (14,), "dim": 768},
        {"grid": (2, 3, 4, 5), "dim": 768},
        {"grid": (14, 0), "dim": 768},
        {"grid": (14, 14), "dim": 0},
        {"grid": (14, 14), "dim": 768, "num_prefix_tokens": -1},
    ],
)
def test_token_layout_rejects_invalid(kwargs):
    with pytest.raises(ValueError):
        TokenLayout(**kwargs)


def test_preprocess_spec_requires_paired_temporal_fields():
    with pytest.raises(ValueError):
        PreprocessSpec(image_size=224, mean=IMAGENET_MEAN, std=IMAGENET_STD, num_frames=16)
    spec = PreprocessSpec(
        image_size=224, mean=IMAGENET_MEAN, std=IMAGENET_STD, num_frames=16, frame_stride=4
    )
    assert spec.num_frames == 16


def test_preprocess_spec_rejects_bad_normalisation():
    with pytest.raises(ValueError):
        PreprocessSpec(image_size=224, mean=(0.5, 0.5), std=IMAGENET_STD)
    with pytest.raises(ValueError):
        PreprocessSpec(image_size=224, mean=IMAGENET_MEAN, std=(0.0, 0.2, 0.2))


# -- forward shapes -------------------------------------------------------


def test_image_encoder_returns_grid_and_prefix():
    enc = DummyImageEncoder()
    out = enc(torch.randn(2, 3, 224, 224))
    assert out.tokens.shape == (2, 196, 768)
    assert out.prefix.shape == (2, 5, 768)
    assert enc.feature_dim == 768


def test_video_encoder_returns_spatiotemporal_grid():
    enc = DummyVideoEncoder()
    out = enc(torch.randn(2, 16, 3, 224, 224))
    assert out.tokens.shape == (2, 1568, 768)
    assert out.prefix is None


def test_tokens_reshape_to_declared_grid():
    enc = DummyVideoEncoder()
    out = enc(torch.randn(1, 16, 3, 224, 224))
    reshaped = out.tokens.reshape(1, *enc.token_layout.grid, enc.feature_dim)
    assert reshaped.shape == (1, 8, 14, 14, 768)


# -- input validation -----------------------------------------------------


def test_rejects_wrong_rank():
    enc = DummyImageEncoder()
    with pytest.raises(ValueError, match=r"\[B, C, H, W\]"):
        enc(torch.randn(16, 3, 224, 224, 1))


def test_video_rejects_channels_time_transposed():
    """[B, C, T, H, W] must be caught rather than silently mis-encoded."""
    enc = DummyVideoEncoder()
    with pytest.raises(ValueError, match="3 input channels"):
        enc(torch.randn(2, 3, 16, 224, 224))


def test_rejects_wrong_resolution():
    enc = DummyImageEncoder()
    with pytest.raises(ValueError, match="preprocess_spec"):
        enc(torch.randn(2, 3, 256, 256))


def test_rejects_wrong_frame_count():
    enc = DummyVideoEncoder()
    with pytest.raises(ValueError, match="8 frames|Expected 16 frames"):
        enc(torch.randn(2, 8, 3, 224, 224))


# -- output validation ----------------------------------------------------


def test_unstripped_prefix_tokens_are_caught():
    enc = LeakyPrefixEncoder()
    with pytest.raises(ValueError, match="prefix tokens were not"):
        enc(torch.randn(1, 3, 224, 224))


def test_layout_declaring_prefix_but_returning_none_is_caught():
    class Mismatched(DummyImageEncoder):
        def _forward_tokens(self, x):
            out = super()._forward_tokens(x)
            return EncoderOutput(tokens=out.tokens, prefix=None)

    with pytest.raises(ValueError, match="returned prefix=None"):
        Mismatched()(torch.randn(1, 3, 224, 224))


# -- freezing and determinism --------------------------------------------


def test_frozen_by_default_and_stays_eval():
    enc = DummyImageEncoder()
    assert enc.is_frozen
    assert not enc.training
    assert all(not p.requires_grad for p in enc.parameters())

    enc.train(True)
    assert not enc.training, "frozen encoder must ignore train(True)"


def test_parent_module_cannot_reactivate_frozen_encoder():
    class Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = DummyImageEncoder()
            self.head = nn.Linear(768, 3)

    model = Wrapper()
    model.train()
    assert model.head.training
    assert not model.encoder.training


def test_unfreeze_restores_normal_mode_switching():
    enc = DummyImageEncoder(freeze=False)
    assert not enc.is_frozen
    enc.train(True)
    assert enc.training


def test_extract_requires_frozen_encoder():
    enc = DummyImageEncoder(freeze=False)
    with pytest.raises(RuntimeError, match="requires a frozen encoder"):
        enc.extract(torch.randn(1, 3, 224, 224))


def test_extract_casts_to_cache_dtype_and_detaches():
    enc = DummyImageEncoder()
    out = enc.extract(torch.randn(2, 3, 224, 224))
    assert out.tokens.dtype == torch.float16
    assert out.prefix.dtype == torch.float16
    assert not out.tokens.requires_grad


def test_extract_is_deterministic():
    enc = DummyImageEncoder()
    x = torch.randn(2, 3, 224, 224)
    a = enc.extract(x, to_dtype=None)
    b = enc.extract(x, to_dtype=None)
    assert torch.equal(a.tokens, b.tokens), "cached features must be reproducible"


def test_mean_pooling_recoverable_from_grid():
    """Pooled vectors are derivable from the cache; the reverse is not."""
    enc = DummyImageEncoder()
    out = enc.extract(torch.randn(2, 3, 224, 224), to_dtype=None)
    pooled = out.tokens.mean(dim=1)
    assert pooled.shape == (2, 768)


# -- provenance -----------------------------------------------------------


def test_describe_captures_protocol():
    enc = DummyImageEncoder()
    d = enc.describe()
    assert d["modality"] == "image"
    assert d["frozen"] is True
    assert d["preprocess"]["image_size"] == 224
    assert d["token_layout"]["grid"] == (14, 14)
    assert d["num_parameters"] > 0


def test_describe_differs_when_protocol_differs():
    """Two arms with different preprocessing must be distinguishable."""
    a = DummyImageEncoder().describe()
    b = DummyVideoEncoder().describe()
    assert a["preprocess"] != b["preprocess"]
    assert a["token_layout"] != b["token_layout"]