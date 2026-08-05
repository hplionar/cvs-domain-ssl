"""Tests for scripts/check_collapse.py."""

from __future__ import annotations

import torch

from scripts.check_collapse import feature_statistics, verdict


def test_healthy_features_span_many_dimensions():
    torch.manual_seed(0)
    stats = feature_statistics(torch.randn(2000, 64))
    assert stats["effective_rank"] > 40
    assert abs(stats["mean_pairwise_cosine"]) < 0.1


def test_collapsed_features_detected():
    """Every patch identical: variance zero, rank one, cosine one."""
    collapsed = torch.ones(500, 64) + torch.randn(500, 64) * 1e-6
    stats = feature_statistics(collapsed)
    assert stats["mean_pairwise_cosine"] > 0.99
    assert stats["effective_rank"] < 5


def test_rank_one_structure_detected():
    """All patches on a single direction, differing only in magnitude."""
    direction = torch.randn(64)
    tokens = torch.randn(500, 1) * direction
    assert feature_statistics(tokens)["effective_rank"] < 2


def test_verdict_flags_variance_collapse():
    before = {"mean_variance": 1.0, "effective_rank": 50.0, "centred_rank": 48.0,
              "dimensions": 64, "rank_fraction": 0.78, "centred_rank_fraction": 0.75,
              "mean_pairwise_cosine": 0.0, "feature_norm": 8.0}
    after = dict(before, mean_variance=0.01)
    status, problems = verdict(before, after)
    assert status == "COLLAPSED"
    assert any("variance" in p for p in problems)


def test_verdict_flags_rank_collapse():
    before = {"mean_variance": 1.0, "effective_rank": 50.0, "centred_rank": 48.0,
              "dimensions": 64, "rank_fraction": 0.78, "centred_rank_fraction": 0.75,
              "mean_pairwise_cosine": 0.0, "feature_norm": 8.0}
    after = dict(before, effective_rank=3.0, rank_fraction=0.05)
    status, problems = verdict(before, after)
    assert status == "COLLAPSED"
    assert any("rank" in p for p in problems)


def test_verdict_flags_identical_patches():
    before = {"mean_variance": 1.0, "effective_rank": 50.0, "centred_rank": 48.0,
              "dimensions": 64, "rank_fraction": 0.78, "centred_rank_fraction": 0.75,
              "mean_pairwise_cosine": 0.1, "feature_norm": 8.0}
    after = dict(before, mean_pairwise_cosine=0.99)
    assert verdict(before, after)[0] == "COLLAPSED"


def test_healthy_change_passes():
    """Adaptation should change the representation without destroying it."""
    before = {"mean_variance": 1.0, "effective_rank": 50.0, "centred_rank": 48.0,
              "dimensions": 64, "rank_fraction": 0.78, "centred_rank_fraction": 0.75,
              "mean_pairwise_cosine": 0.05, "feature_norm": 8.0}
    after = {"mean_variance": 0.8, "effective_rank": 42.0, "centred_rank": 41.0,
             "dimensions": 64, "rank_fraction": 0.66, "centred_rank_fraction": 0.64,
             "mean_pairwise_cosine": 0.12, "feature_norm": 7.2}
    status, problems = verdict(before, after)
    assert status == "healthy"
    assert problems == []


def test_dimensions_reported():
    assert feature_statistics(torch.randn(100, 1024))["dimensions"] == 1024


# -- the two ranks answer different questions -----------------------------


def test_constant_offset_depresses_uncentred_but_not_centred():
    """A large shared offset across patches dominates the uncentred spectrum
    even where variation is high-dimensional. Reading the uncentred figure as
    capacity usage understates it by an order of magnitude: VideoMAE ViT-B on
    surgical video measures 2.6 uncentred against 31.9 centred."""
    torch.manual_seed(0)
    variation = torch.randn(2000, 64)
    offset = torch.randn(64) * 30.0
    stats = feature_statistics(variation + offset)

    assert stats["effective_rank"] < 5, "offset should dominate the uncentred spectrum"
    assert stats["centred_rank"] > 40, "variation is high-dimensional once centred"


def test_true_collapse_is_low_on_both():
    """Genuine collapse leaves nothing in either measure."""
    stats = feature_statistics(torch.ones(500, 64) + torch.randn(500, 64) * 1e-6)
    assert stats["effective_rank"] < 2
    assert stats["mean_pairwise_cosine"] > 0.99


def test_centred_rank_fraction_reported():
    stats = feature_statistics(torch.randn(500, 128))
    assert 0.0 < stats["centred_rank_fraction"] <= 1.0


def test_verdict_flags_lost_variation():
    """Scale can shrink harmlessly; losing dimensions of variation cannot."""
    before = {"mean_variance": 1.0, "effective_rank": 50.0, "centred_rank": 48.0,
              "dimensions": 64, "rank_fraction": 0.78, "centred_rank_fraction": 0.75,
              "mean_pairwise_cosine": 0.1, "feature_norm": 8.0}
    after = dict(before, centred_rank=5.0, centred_rank_fraction=0.08)
    status, problems = verdict(before, after)
    assert status == "COLLAPSED"
    assert any("centred rank" in p for p in problems)


def test_verdict_tolerates_modest_change_in_both():
    before = {"mean_variance": 1.0, "effective_rank": 50.0, "centred_rank": 48.0,
              "dimensions": 64, "rank_fraction": 0.78, "centred_rank_fraction": 0.75,
              "mean_pairwise_cosine": 0.05, "feature_norm": 8.0}
    after = {"mean_variance": 0.9, "effective_rank": 46.0, "centred_rank": 44.0,
             "dimensions": 64, "rank_fraction": 0.72, "centred_rank_fraction": 0.69,
             "mean_pairwise_cosine": 0.08, "feature_norm": 7.6}
    assert verdict(before, after)[0] == "healthy"