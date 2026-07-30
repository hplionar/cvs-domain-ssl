"""Tests for the encoder registry."""

from __future__ import annotations

import pytest
import torch

from models.encoders import (
    BaseEncoder,
    EncoderOutput,
    PreprocessSpec,
    TokenLayout,
    available_encoders,
    build_encoder,
    register_encoder,
)


class _Tiny(BaseEncoder):
    modality = "image"

    def __init__(self, *, freeze: bool = True, dim: int = 8) -> None:
        super().__init__(freeze=freeze)
        self._dim = dim
        self.scale = torch.nn.Parameter(torch.ones(dim))
        self._finalise_init()

    @property
    def preprocess_spec(self):
        return PreprocessSpec(image_size=32, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

    @property
    def token_layout(self):
        return TokenLayout(grid=(2, 2), dim=self._dim)

    def _forward_tokens(self, x):
        b = x.shape[0]
        return EncoderOutput(tokens=torch.ones(b, 4, self._dim) * self.scale)


def test_register_build_and_list():
    register_encoder("tiny_test")(_Tiny)
    assert "tiny_test" in available_encoders()
    enc = build_encoder("TINY_TEST", dim=8)
    assert isinstance(enc, BaseEncoder)
    assert enc(torch.randn(2, 3, 32, 32)).tokens.shape == (2, 4, 8)


def test_kwargs_reach_the_factory():
    register_encoder("tiny_dim16")(_Tiny)
    assert build_encoder("tiny_dim16", dim=16).feature_dim == 16


def test_unknown_encoder_lists_alternatives():
    with pytest.raises(KeyError, match="Available"):
        build_encoder("does_not_exist")


def test_duplicate_registration_rejected():
    register_encoder("tiny_dupe")(_Tiny)
    with pytest.raises(ValueError, match="already registered"):
        register_encoder("tiny_dupe")(_Tiny)


def test_non_conforming_factory_rejected():
    register_encoder("not_an_encoder")(lambda **kw: torch.nn.Linear(1, 1))
    with pytest.raises(TypeError, match="does not implement BaseEncoder"):
        build_encoder("not_an_encoder")