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
    num_prefix: int = 0,
    prefix_signal: bool = False,
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

    if num_prefix:
        prefix = rng.standard_normal((n, num_prefix, dim)).astype(np.float16)
        if prefix_signal:
            # Put the decodable signal in the CLS token *only*, so a run that
            # succeeds proves the global branch was actually read.
            prefix[:, 0, 0] = targets[:, 0] * 4 - 2
            prefix[:, 0, 1] = targets[:, 1] * 4 - 2
            prefix[:, 0, 2] = targets[:, 2] * 4 - 2
        np.save(path / "prefix.npy", prefix)
    with open(path / "index.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["sample_id", "video_id", "row"])
        for i in range(n):
            w.writerow([f"s{i:04d}", f"v{i // 8:02d}", i])

    manifest = {
        "encoder": {
            "checkpoint_id": checkpoint_id,
            "token_layout": {"grid": [2, 2], "dim": dim, "num_prefix_tokens": num_prefix},
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
    tokens, prefix, target = cache[0]
    assert tokens.shape == (4, 8)
    assert tokens.dtype == torch.float32, "fp16 cache must be cast for training"
    assert target.shape == (3,)
    assert not cache.has_prefix
    assert prefix.shape == (0, 8), (
        "a cache without prefix tokens must still yield a collatable placeholder"
    )


# -- prefix tokens --------------------------------------------------------


def test_loads_prefix_when_present(tmp_path):
    """extract_features.py writes prefix.npy; the trainer must actually read it.

    Before the fusion head existed, CachedFeatures ignored prefix.npy entirely,
    so the global branch input was on disk and unreachable.
    """
    cache = CachedFeatures(make_cache(tmp_path / "c", num_prefix=2))
    assert cache.has_prefix
    assert cache.num_prefix_tokens == 2
    _, prefix, _ = cache[0]
    assert prefix.shape == (2, 8)
    assert prefix.dtype == torch.float32


def test_prefix_rows_align_with_token_rows(tmp_path):
    """Row i of prefix.npy must describe the same sample as row i of tokens.npy.

    Video-level subsetting reindexes through self.rows; if prefix were indexed
    by the dense position instead, every learning-curve point would silently
    pair a sample's patches with another sample's CLS token.
    """
    path = make_cache(tmp_path / "c", n=32, num_prefix=1)
    full = CachedFeatures(path)
    subset = CachedFeatures(path, video_ids=["v01"])  # rows 8..15

    for local, row in enumerate(range(8, 16)):
        assert torch.equal(subset[local][1], full[row][1])
        assert torch.equal(subset[local][0], full[row][0])


def test_inconsistent_prefix_row_count_is_rejected(tmp_path):
    path = make_cache(tmp_path / "c", n=32, num_prefix=1)
    np.save(path / "prefix.npy", np.zeros((31, 1, 8), dtype=np.float16))
    with pytest.raises(ValueError, match="inconsistent"):
        CachedFeatures(path)


def test_inconsistent_prefix_dim_is_rejected(tmp_path):
    path = make_cache(tmp_path / "c", n=32, num_prefix=1)
    np.save(path / "prefix.npy", np.zeros((32, 1, 16), dtype=np.float16))
    with pytest.raises(ValueError, match="does not match token dim"):
        CachedFeatures(path)


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
    tokens, prefix, _ = ds[0]
    assert tokens.shape == (1, 8), "heads expect [N, D] even when N == 1"
    assert prefix.shape == (0, 8)


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


def _run(tmp_path, cache_kwargs=None, **overrides):
    import argparse

    from train.train_probe_cached import main
    import sys

    cache_kwargs = cache_kwargs or {}
    train = make_cache(tmp_path / "train", n=64, signal=True, **cache_kwargs)
    val = make_cache(tmp_path / "val", n=32, signal=True, **cache_kwargs)
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


def test_fusion_head_path_runs_with_cls(tmp_path):
    out = _run(
        tmp_path,
        cache_kwargs={"num_prefix": 1},
        **{"--head": "fusion", "--attn-hidden": 16},
    )
    results = json.loads((out / "results.json").read_text())
    assert results["head"]["kind"] == "fusion"
    assert results["head"]["global_source"] == "cls"
    assert results["head"]["pooling_precomputed"] is False
    cfg = results["head"]["selected_head_config"]
    assert cfg["uses_prefix"] is True
    assert cfg["fused_dim"] == 2 * cfg["input_dim"]


def test_fusion_head_reads_the_cls_token(tmp_path):
    """A cache whose only decodable signal is in [CLS] must still be learnable.

    Without this, a fusion run that quietly ignored prefix.npy would look
    healthy: the MIL branch alone would fit the patch-token signal that
    ``signal=True`` also plants, and the global branch could be dead.
    """
    import sys

    from train.train_probe_cached import main

    train = make_cache(tmp_path / "train", n=96, num_prefix=1, prefix_signal=True)
    val = make_cache(tmp_path / "val", n=48, num_prefix=1, prefix_signal=True)
    out = tmp_path / "out"
    argv = [
        "prog",
        "--train-features", str(train), "--val-features", str(val),
        "--output-dir", str(out), "--epochs", "40", "--seeds", "1",
        "--lr", "1e-2", "--weight-decay", "0.0", "--dropout", "0.0",
        "--attn-hidden", "16", "--device", "cpu", "--patience", "0",
        "--head", "fusion",
    ]
    old, sys.argv = sys.argv, argv
    try:
        assert main() == 0
    finally:
        sys.argv = old

    results = json.loads((out / "results.json").read_text())
    assert results["selected"]["mean_map"] > 0.75, (
        "signal present only in [CLS] was not learned; the global branch is not "
        "receiving prefix tokens"
    )


def test_fusion_head_falls_back_only_when_told(tmp_path):
    """No prefix in the cache: 'auto' resolves to patch_mean and records it,
    but an explicit --global-source cls fails loudly rather than degrading."""
    out = _run(tmp_path, **{"--head": "fusion", "--attn-hidden": 16})
    results = json.loads((out / "results.json").read_text())
    assert results["head"]["global_source"] == "patch_mean"
    assert results["head"]["selected_head_config"]["uses_prefix"] is False

    with pytest.raises(SystemExit, match="requires prefix tokens"):
        _run(tmp_path / "b", **{"--head": "fusion", "--global-source": "cls"})


def test_mismatched_prefix_availability_between_splits_is_rejected(tmp_path):
    """Declared prefix counts differ, so verify_protocol stops the run first.

    The manifest's token_layout already carries num_prefix_tokens, so this is
    caught before any head is built. resolve_global_source repeats the check on
    what is actually on disk, which is a different question: a manifest can
    declare prefix tokens that prefix.npy does not contain.
    """
    import sys

    from train.train_probe_cached import main

    train = make_cache(tmp_path / "train", n=32, signal=True, num_prefix=1)
    val = make_cache(tmp_path / "val", n=32, signal=True, num_prefix=0)
    argv = [
        "prog", "--train-features", str(train), "--val-features", str(val),
        "--output-dir", str(tmp_path / "out"), "--epochs", "2", "--seeds", "1",
        "--lr", "1e-2", "--weight-decay", "0.0", "--dropout", "0.0",
        "--device", "cpu", "--head", "fusion",
    ]
    old, sys.argv = sys.argv, argv
    try:
        with pytest.raises(ValueError, match="not comparable"):
            main()
    finally:
        sys.argv = old


def test_resolve_global_source_checks_files_not_just_manifests(tmp_path):
    """A manifest that declares prefix tokens the cache does not hold is caught.

    Deleting prefix.npy leaves both manifests identical, so verify_protocol
    passes; only an on-disk check finds it.
    """
    import argparse

    from train.train_probe_cached import resolve_global_source

    train = CachedFeatures(make_cache(tmp_path / "train", n=16, num_prefix=1))
    val_path = make_cache(tmp_path / "val", n=16, num_prefix=1)
    (val_path / "prefix.npy").unlink()
    val = CachedFeatures(val_path)

    args = argparse.Namespace(head="fusion", global_source="auto")
    with pytest.raises(SystemExit, match="same protocol"):
        resolve_global_source(args, train, val)


def test_resolve_global_source_is_inert_for_other_heads(tmp_path):
    import argparse

    from train.train_probe_cached import resolve_global_source

    cache = CachedFeatures(make_cache(tmp_path / "c", n=16))
    args = argparse.Namespace(head="attentive", global_source="auto")
    assert resolve_global_source(args, cache, cache) == "auto"


def test_attention_grid_adds_hidden_dim_only_for_attention_heads():
    """Equal search effort per arm is the protocol. hidden_dim names nothing in
    the mean head, so adding it there would double that arm's runs for nothing
    and make the two grids incomparable in size."""
    import argparse

    args = argparse.Namespace(lr=None, weight_decay=None, dropout=None, attn_hidden=None)
    assert len(build_grid(args, attention=False)) == 4 * 3 * 2
    attn = build_grid(args, attention=True)
    assert len(attn) == 4 * 3 * 2 * 2
    assert {c["hidden_dim"] for c in attn} == {128, 512}
    assert all("hidden_dim" not in c for c in build_grid(args, attention=False))


def test_attention_grid_is_overridable():
    import argparse

    args = argparse.Namespace(
        lr=[1e-3], weight_decay=[0.0], dropout=[0.0], attn_hidden=[64]
    )
    assert build_grid(args, attention=True) == [
        {"dropout": 0.0, "hidden_dim": 64, "lr": 1e-3, "weight_decay": 0.0}
    ]


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