"""Tests for train/train_probe_cached.py.

Caches are synthesised directly rather than produced by running an encoder, so
these run offline in seconds.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from train.train_probe_cached import (
    CachedFeatures,
    PooledFeatures,
    build_grid,
    compute_pos_weight,
    verify_protocol,
    verify_same_encoder,
)


def make_cache(
    path: Path,
    *,
    n: int = 32,
    tokens: int = 4,
    dim: int = 8,
    checkpoint_id: str = "enc-a",
    image_size: int = 224,
    reduction: str = "none",
    signal: bool = False,
) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    feats = rng.standard_normal((n, tokens, dim)).astype(np.float16)
    targets = rng.integers(0, 2, size=(n, 3)).astype(np.float32)
    if signal:
        # make the label linearly decodable so training can be verified
        feats[:, 0, 0] = targets[:, 0] * 4 - 2
        feats[:, 0, 1] = targets[:, 1] * 4 - 2
        feats[:, 0, 2] = targets[:, 2] * 4 - 2

    np.save(path / "tokens.npy", feats)
    np.save(path / "targets.npy", targets)
    with open(path / "index.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["sample_id", "video_id", "row"])
        for i in range(n):
            w.writerow([f"s{i:04d}", f"v{i // 8:02d}", i])

    manifest = {
        "encoder": {
            "checkpoint_id": checkpoint_id,
            "token_layout": {"grid": [2, 2], "dim": dim, "num_prefix_tokens": 0},
        },
        "transform": {"image_size": image_size, "mean": [0.485, 0.456, 0.406]},
        "extraction": {"cache_dtype": "float16", "reduction": reduction},
    }
    (path / "manifest.json").write_text(json.dumps(manifest))
    return path


# -- cache loading --------------------------------------------------------


def test_loads_cache(tmp_path):
    cache = CachedFeatures(make_cache(tmp_path / "c"))
    assert len(cache) == 32
    assert cache.feature_dim == 8
    assert cache.num_tokens == 4
    tokens, target = cache[0]
    assert tokens.shape == (4, 8)
    assert tokens.dtype == torch.float32, "fp16 cache must be cast for training"
    assert target.shape == (3,)


def test_missing_manifest_is_rejected(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="No manifest.json"):
        CachedFeatures(tmp_path / "empty")


def test_video_level_subset(tmp_path):
    """Learning curves must subsample by video, not by frame: frames within a
    procedure are strongly correlated."""
    cache = CachedFeatures(make_cache(tmp_path / "c"), video_ids=["v00", "v01"])
    assert len(cache) == 16
    assert cache.unique_video_ids() == ["v00", "v01"]


def test_empty_subset_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="zero samples"):
        CachedFeatures(make_cache(tmp_path / "c"), video_ids=["nonexistent"])


def test_precomputed_pooling_matches_per_batch_pooling(tmp_path):
    cache = CachedFeatures(make_cache(tmp_path / "c"))
    pooled = cache.pooled()
    manual = torch.stack([cache[i][0].mean(dim=0) for i in range(len(cache))])
    assert torch.allclose(pooled, manual, atol=1e-5)


def test_pooled_dataset_restores_grid_axis(tmp_path):
    cache = CachedFeatures(make_cache(tmp_path / "c"))
    ds = PooledFeatures(cache.pooled(), cache.all_targets())
    tokens, _ = ds[0]
    assert tokens.shape == (1, 8), "heads expect [N, D] even when N == 1"


# -- protocol verification ------------------------------------------------


def test_identical_protocol_passes(tmp_path):
    a = CachedFeatures(make_cache(tmp_path / "a")).manifest
    b = CachedFeatures(make_cache(tmp_path / "b")).manifest
    verify_protocol(a, b, label_a="a", label_b="b")


def test_mismatched_transform_is_rejected(tmp_path):
    a = CachedFeatures(make_cache(tmp_path / "a", image_size=224)).manifest
    b = CachedFeatures(make_cache(tmp_path / "b", image_size=256)).manifest
    with pytest.raises(ValueError, match="not comparable"):
        verify_protocol(a, b, label_a="a", label_b="b")


def test_mismatched_reduction_is_rejected(tmp_path):
    a = CachedFeatures(make_cache(tmp_path / "a", reduction="none")).manifest
    b = CachedFeatures(make_cache(tmp_path / "b", reduction="spatial")).manifest
    with pytest.raises(ValueError, match="not comparable"):
        verify_protocol(a, b, label_a="a", label_b="b")


def test_mismatched_encoder_between_train_and_val_is_rejected(tmp_path):
    a = CachedFeatures(make_cache(tmp_path / "a", checkpoint_id="enc-a")).manifest
    b = CachedFeatures(make_cache(tmp_path / "b", checkpoint_id="enc-b")).manifest
    with pytest.raises(ValueError, match="but validation cache used"):
        verify_same_encoder(a, b)


def test_different_encoders_may_share_a_protocol(tmp_path):
    """Comparing two arms is the point; only the protocol must match."""
    a = CachedFeatures(make_cache(tmp_path / "a", checkpoint_id="videomae")).manifest
    b = CachedFeatures(make_cache(tmp_path / "b", checkpoint_id="vjepa2")).manifest
    verify_protocol(a, b, label_a="videomae", label_b="vjepa2")


# -- grid and loss --------------------------------------------------------


def test_default_grid_size():
    import argparse

    args = argparse.Namespace(lr=None, weight_decay=None, dropout=None)
    assert len(build_grid(args)) == 4 * 3 * 2


def test_grid_override():
    import argparse

    args = argparse.Namespace(lr=[1e-3], weight_decay=[0.0], dropout=[0.0])
    grid = build_grid(args)
    assert grid == [{"dropout": 0.0, "lr": 1e-3, "weight_decay": 0.0}]


def test_pos_weight_is_negatives_over_positives():
    targets = torch.tensor([[1.0, 0.0, 0.0]] * 2 + [[0.0, 1.0, 1.0]] * 6)
    weights = compute_pos_weight(targets)
    assert torch.allclose(weights, torch.tensor([3.0, 1.0 / 3.0, 1.0 / 3.0]))


def test_pos_weight_handles_absent_class():
    targets = torch.zeros(4, 3)
    assert torch.isfinite(compute_pos_weight(targets)).all()


# -- end to end -----------------------------------------------------------


def _run(tmp_path, **overrides):
    import argparse

    from train.train_probe_cached import main
    import sys

    train = make_cache(tmp_path / "train", n=64, signal=True)
    val = make_cache(tmp_path / "val", n=32, signal=True)
    out = tmp_path / "out"
    argv = [
        "prog",
        "--train-features", str(train),
        "--val-features", str(val),
        "--output-dir", str(out),
        "--epochs", "8",
        "--seeds", "2",
        "--lr", "1e-2",
        "--weight-decay", "0.0",
        "--dropout", "0.0",
        "--device", "cpu",
        "--patience", "0",
    ]
    for k, v in overrides.items():
        argv += [k, *([] if v is None else [str(v)])]
    old = sys.argv
    sys.argv = argv
    try:
        assert main() == 0
    finally:
        sys.argv = old
    return out


def test_end_to_end_writes_results(tmp_path):
    out = _run(tmp_path)
    results = json.loads((out / "results.json").read_text())
    assert "selected" in results
    assert results["selected"]["mean_map"] > 0.5, "linearly decodable signal not learned"
    assert len(results["selected"]["seeds"]) == 2
    assert results["head"]["pooling_precomputed"] is True


def test_end_to_end_saves_logits_for_threshold_tuning(tmp_path):
    out = _run(tmp_path)
    files = sorted(out.glob("val_logits_seed*.npz"))
    assert len(files) == 2
    payload = np.load(files[0])
    assert payload["logits"].shape == (32, 3)
    assert payload["targets"].shape == (32, 3)


def test_end_to_end_writes_history_per_seed(tmp_path):
    out = _run(tmp_path)
    histories = sorted(out.glob("history_seed*.json"))
    assert len(histories) == 2
    rows = json.loads(histories[0].read_text())
    assert "mAP" in rows[0] and "train_loss" in rows[0]


def test_attentive_head_path_runs(tmp_path):
    out = _run(tmp_path, **{"--head": "attentive"})
    results = json.loads((out / "results.json").read_text())
    assert results["head"]["kind"] == "attentive"
    assert results["head"]["pooling_precomputed"] is False


def test_seed_variance_is_reported(tmp_path):
    out = _run(tmp_path)
    results = json.loads((out / "results.json").read_text())
    assert "std_map" in results["selected"]


# -- learning-curve selection ---------------------------------------------


def test_learning_curve_selects_on_seed_mean_not_best_run(tmp_path, monkeypatch):
    """Selecting the maximum over the grid biases every point upward, because
    the maximum of several noisy estimates exceeds the mean of the underlying
    quantities. The bias grows with variance, and variance is largest at small
    training fractions -- so the curve would be inflated most at exactly the end
    where an advantage for the adapted encoder is being claimed."""
    import argparse

    from train.train_probe_cached import RunResult, aggregate

    grid = [{"lr": 1e-3}, {"lr": 3e-3}]

    # Config A is better on average; config B has one lucky seed that is the
    # single highest number present.
    results = [
        RunResult(config=grid[0], seed=0, best_epoch=1, best_map=0.50, best_metrics={}),
        RunResult(config=grid[0], seed=1, best_epoch=1, best_map=0.52, best_metrics={}),
        RunResult(config=grid[1], seed=0, best_epoch=1, best_map=0.61, best_metrics={}),
        RunResult(config=grid[1], seed=1, best_epoch=1, best_map=0.30, best_metrics={}),
    ]

    ranked = aggregate(results, grid)
    assert ranked[0]["config"] == grid[0], "should prefer the better mean"
    assert ranked[0]["mean_map"] == pytest.approx(0.51)

    lucky = max(results, key=lambda r: r.best_map)
    assert lucky.config == grid[1]
    assert lucky.best_map > ranked[0]["mean_map"], (
        "the single best run exceeds the selected mean; selecting on it would "
        "inflate the reported point"
    )


def test_learning_curve_records_the_selected_config(tmp_path):
    """The curve must show which configuration produced each point, so a reader
    can tell whether the shape reflects data quantity or a changing recipe."""
    import inspect

    from train.train_probe_cached import learning_curve

    source = inspect.getsource(learning_curve)
    assert "aggregate(" in source, "selection should go through aggregate()"
    assert "selected_config" in source
    assert "key=lambda r: r.best_map" not in source, (
        "max-over-grid selection reintroduces the upward bias"
    )