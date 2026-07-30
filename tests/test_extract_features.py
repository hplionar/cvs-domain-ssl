"""Tests for scripts/extract_features.py.

All run offline against randomly initialised encoders.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pytest
import torch

pytest.importorskip("transformers")
pytest.importorskip("torchvision")

from scripts.extract_features import (  # noqa: E402
    SyntheticDataset,
    apply_reduction,
    extract,
    reduced_token_count,
)
from models.encoders.videomae_encoder import VideoMAEEncoder  # noqa: E402
from models.encoders.mae_encoder import MAEEncoder  # noqa: E402

TINY = dict(hidden_size=32, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32)


def _args(**overrides) -> argparse.Namespace:
    base = dict(
        encoder="mae_b", model_name=None, dataset="endoscapes", split="train",
        dataset_root=None, manifest_path=None, out=None, reduction="none",
        batch_size=4, num_workers=0, prefetch_factor=2, device="cpu", amp=False,
        log_every=0, smoke=True, smoke_size=8, dry_run=False, random_init=True,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


@pytest.fixture(scope="module")
def videomae():
    from transformers import VideoMAEConfig, VideoMAEModel

    cfg = VideoMAEConfig(image_size=224, patch_size=16, num_frames=16, tubelet_size=2, **TINY)
    return VideoMAEEncoder(model=VideoMAEModel(cfg))


@pytest.fixture(scope="module")
def mae():
    from transformers import ViTMAEConfig, ViTMAEModel

    return MAEEncoder(model=ViTMAEModel(ViTMAEConfig(image_size=224, patch_size=16, **TINY)))


# -- reduction ------------------------------------------------------------


def test_reduction_none_is_identity(videomae):
    tokens = torch.randn(2, 1568, 32)
    assert torch.equal(apply_reduction(tokens, videomae, "none"), tokens)
    assert reduced_token_count(videomae, "none") == 1568


def test_spatial_reduction_keeps_temporal_axis(videomae):
    tokens = torch.randn(2, 1568, 32)
    out = apply_reduction(tokens, videomae, "spatial")
    assert out.shape == (2, 8, 32)
    assert reduced_token_count(videomae, "spatial") == 8


def test_spatial_reduction_matches_manual_pooling(videomae):
    tokens = torch.randn(1, 1568, 32)
    expected = tokens.reshape(1, 8, 196, 32).mean(dim=2)
    assert torch.allclose(apply_reduction(tokens, videomae, "spatial"), expected)


def test_full_reduction_collapses_to_one_token(videomae, mae):
    assert apply_reduction(torch.randn(2, 1568, 32), videomae, "full").shape == (2, 1, 32)
    assert reduced_token_count(mae, "full") == 1


def test_spatial_reduction_on_image_encoder_collapses(mae):
    """An image encoder has no temporal axis, so spatial == full."""
    assert reduced_token_count(mae, "spatial") == 1


# -- synthetic dataset ----------------------------------------------------


def test_synthetic_shapes_match_encoder_spec(videomae, mae):
    assert SyntheticDataset(mae, 4)[0]["image"].shape == (3, 224, 224)
    assert SyntheticDataset(videomae, 4)[0]["image"].shape == (16, 3, 224, 224)


def test_synthetic_dataset_is_reproducible(mae):
    a, b = SyntheticDataset(mae, 4), SyntheticDataset(mae, 4)
    assert torch.equal(a[2]["image"], b[2]["image"])


# -- end to end -----------------------------------------------------------


def test_extraction_writes_complete_cache(tmp_path):
    out = tmp_path / "cache"
    assert extract(_args(out=str(out), smoke_size=8, batch_size=4)) == 0

    tokens = np.load(out / "tokens.npy", mmap_mode="r")
    assert tokens.shape == (8, 196, 768)
    assert tokens.dtype == np.float16
    assert np.isfinite(np.asarray(tokens)).all()

    prefix = np.load(out / "prefix.npy", mmap_mode="r")
    assert prefix.shape == (8, 1, 768)

    assert np.load(out / "targets.npy").shape == (8, 3)
    assert sum(1 for _ in open(out / "index.csv")) == 9  # header + 8


def test_manifest_records_protocol(tmp_path):
    out = tmp_path / "cache"
    extract(_args(out=str(out), smoke_size=4, batch_size=2))
    m = json.loads((out / "manifest.json").read_text())

    assert m["encoder"]["token_layout"]["grid"] == [14, 14]
    assert m["transform"]["image_size"] == 224
    assert m["transform"]["mean"] == [0.485, 0.456, 0.406]
    assert m["extraction"]["cache_dtype"] == "float16"
    assert m["num_samples"] == 4
    assert "torch" in m["environment"]


def test_reduction_changes_cache_shape(tmp_path):
    out = tmp_path / "reduced"
    extract(_args(out=str(out), smoke_size=4, batch_size=2, reduction="full"))
    assert np.load(out / "tokens.npy", mmap_mode="r").shape == (4, 1, 768)
    assert not (out / "prefix.npy").exists(), "prefix is meaningless once pooled"


def test_index_rows_are_ordered_and_unique(tmp_path):
    import csv as _csv

    out = tmp_path / "cache"
    extract(_args(out=str(out), smoke_size=8, batch_size=3))
    with open(out / "index.csv") as fh:
        rows = list(_csv.DictReader(fh))
    assert [int(r["row"]) for r in rows] == list(range(8))
    assert len({r["sample_id"] for r in rows}) == 8


def test_dry_run_writes_nothing(tmp_path):
    out = tmp_path / "nothing"
    assert extract(_args(out=str(out), dry_run=True)) == 0
    assert not (out / "tokens.npy").exists()


def test_video_encoder_rejects_frame_dataset():
    with pytest.raises(ValueError, match="video model"):
        extract(_args(encoder="videomae_b", dataset="endoscapes", smoke=False,
                      dataset_root="/nonexistent", manifest_path="/nonexistent",
                      out="/tmp/never"))