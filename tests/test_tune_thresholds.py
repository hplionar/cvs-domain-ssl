"""Tests for eval/tune_thresholds.py."""

from __future__ import annotations

import json

import numpy as np
import pytest

from eval.tune_thresholds import (
    apply_thresholds,
    load_seed_logits,
    summarise_run,
    tune_all_criteria,
    tune_threshold,
)


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


# -- single criterion -----------------------------------------------------


def test_tuning_finds_a_separating_threshold():
    rng = np.random.default_rng(0)
    y = np.concatenate([np.zeros(100), np.ones(100)]).astype(int)
    p = np.concatenate([rng.uniform(0.0, 0.4, 100), rng.uniform(0.6, 1.0, 100)])
    threshold, score = tune_threshold(y, p)
    assert 0.4 <= threshold <= 0.65
    assert score > 0.95


def test_tuning_beats_fixed_cutoff_on_shifted_scores():
    """The F3 mechanism in miniature: a well-ranked but miscalibrated model."""
    from sklearn.metrics import balanced_accuracy_score

    y = np.concatenate([np.zeros(100), np.ones(100)]).astype(int)
    p = np.concatenate([np.full(100, 0.80), np.full(100, 0.95)])  # all above 0.5
    at_half = balanced_accuracy_score(y, (p >= 0.5).astype(int))
    _, tuned = tune_threshold(y, p)
    assert at_half == 0.5, "fixed cutoff calls everything positive"
    assert tuned == 1.0, "tuning recovers perfect separation from identical ranking"


def test_single_class_returns_default_threshold():
    threshold, score = tune_threshold(np.zeros(10, dtype=int), np.random.rand(10))
    assert threshold == 0.5
    assert np.isnan(score)


def test_quantile_candidates_handle_narrow_score_bands():
    """An even grid on [0,1] would have almost no resolution here."""
    y = np.concatenate([np.zeros(50), np.ones(50)]).astype(int)
    p = np.concatenate([np.full(50, 0.4990), np.full(50, 0.4995)])
    _, score = tune_threshold(y, p)
    assert score == 1.0


# -- all criteria ---------------------------------------------------------


def test_tune_all_criteria_reports_both_operating_points():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, size=(200, 3)).astype(int)
    result = tune_all_criteria(y, rng.standard_normal((200, 3)))
    assert set(result["thresholds"]) == {"c1", "c2", "c3"}
    assert "mean_bacc_tuned" in result and "mean_bacc_at_0.5" in result


def test_tuned_is_never_worse_than_fixed_on_the_tuning_split():
    """Tuning maximises the metric on the split it is tuned on, by construction."""
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, size=(300, 3)).astype(int)
    scores = rng.standard_normal((300, 3)) + y * 0.8
    result = tune_all_criteria(y, scores)
    assert result["mean_bacc_tuned"] >= result["mean_bacc_at_0.5"] - 1e-9


def test_criteria_get_independent_thresholds():
    """Different prevalences imply different optimal cutoffs."""
    rng = np.random.default_rng(3)
    n = 400
    y = np.stack([
        (rng.random(n) < 0.5).astype(int),
        (rng.random(n) < 0.1).astype(int),
        (rng.random(n) < 0.9).astype(int),
    ], axis=1)
    scores = rng.standard_normal((n, 3)) + y * 1.5
    thresholds = tune_all_criteria(y, scores)["thresholds"]
    assert len({round(v, 3) for v in thresholds.values()}) > 1


# -- applying fixed thresholds -------------------------------------------


def test_apply_thresholds_uses_supplied_values():
    y = np.concatenate([np.zeros((50, 3)), np.ones((50, 3))]).astype(int)
    scores = logit(np.concatenate([np.full((50, 3), 0.80), np.full((50, 3), 0.95)]))
    generous = apply_thresholds(y, scores, {"c1": 0.5, "c2": 0.5, "c3": 0.5})
    tuned = apply_thresholds(y, scores, {"c1": 0.9, "c2": 0.9, "c3": 0.9})
    assert generous["mean_bacc"] == 0.5
    assert tuned["mean_bacc"] == 1.0


# -- run directories ------------------------------------------------------


def _write_run(tmp_path, name, *, offset=0.0, seeds=3):
    run = tmp_path / name
    run.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(abs(hash(name)) % 2**32)
    for seed in range(seeds):
        y = rng.integers(0, 2, size=(150, 3)).astype(np.float32)
        base = rng.standard_normal((150, 3)) + y * 1.2
        np.savez(run / f"val_logits_seed{seed}.npz", logits=base + offset, targets=y)
    return run


def test_load_seed_logits(tmp_path):
    payloads = load_seed_logits(_write_run(tmp_path, "run"))
    assert sorted(payloads) == [0, 1, 2]


def test_missing_logits_raises(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="val_logits_seed"):
        load_seed_logits(tmp_path / "empty")


def test_summarise_reports_seed_variance(tmp_path):
    summary = summarise_run(_write_run(tmp_path, "run"))
    assert summary["num_seeds"] == 3
    assert summary["mean_bacc_tuned"]["std"] >= 0
    assert summary["gain_from_tuning"] >= -1e-9


def test_offset_run_recovers_after_tuning(tmp_path):
    """Two runs identical up to a calibration shift should differ at 0.5 and
    agree once each is evaluated at its own tuned threshold. This is exactly
    the F3 situation."""
    unshifted = summarise_run(_write_run(tmp_path, "a", offset=0.0))
    shifted = summarise_run(_write_run(tmp_path, "a_shift", offset=2.5))

    fixed_gap = abs(shifted["mean_bacc_at_0.5"]["mean"] - unshifted["mean_bacc_at_0.5"]["mean"])
    tuned_gap = abs(shifted["mean_bacc_tuned"]["mean"] - unshifted["mean_bacc_tuned"]["mean"])
    assert tuned_gap < fixed_gap, (
        "a pure calibration shift should shrink once both are tuned"
    )


def test_cli_writes_thresholds(tmp_path):
    import sys

    from eval.tune_thresholds import main

    run = _write_run(tmp_path, "run")
    argv = ["prog", "--run-dir", str(run)]
    old, sys.argv = sys.argv, argv
    try:
        assert main() == 0
    finally:
        sys.argv = old

    payload = json.loads((run / "thresholds.json").read_text())
    assert "primary" in payload
    assert payload["primary"]["num_seeds"] == 3


def test_cli_comparison_quantifies_artefact(tmp_path):
    import sys

    from eval.tune_thresholds import main

    a = _write_run(tmp_path, "a", offset=0.0)
    b = _write_run(tmp_path, "a_shift", offset=2.5)
    out = tmp_path / "cmp.json"
    argv = ["prog", "--run-dir", str(a), "--compare-run-dir", str(b), "--output", str(out)]
    old, sys.argv = sys.argv, argv
    try:
        assert main() == 0
    finally:
        sys.argv = old

    payload = json.loads(out.read_text())
    assert "comparison" in payload
    assert "artefact_fraction" in payload["comparison"]