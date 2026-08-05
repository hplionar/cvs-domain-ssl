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
    before = {"mean_variance": 1.0, "effective_rank": 50.0, "dimensions": 64,
              "rank_fraction": 0.78, "mean_pairwise_cosine": 0.0, "feature_norm": 8.0}
    after = dict(before, mean_variance=0.01)
    status, problems = verdict(before, after)
    assert status == "COLLAPSED"
    assert any("variance" in p for p in problems)


def test_verdict_flags_rank_collapse():
    before = {"mean_variance": 1.0, "effective_rank": 50.0, "dimensions": 64,
              "rank_fraction": 0.78, "mean_pairwise_cosine": 0.0, "feature_norm": 8.0}
    after = dict(before, effective_rank=3.0, rank_fraction=0.05)
    status, problems = verdict(before, after)
    assert status == "COLLAPSED"
    assert any("rank" in p for p in problems)


def test_verdict_flags_identical_patches():
    before = {"mean_variance": 1.0, "effective_rank": 50.0, "dimensions": 64,
              "rank_fraction": 0.78, "mean_pairwise_cosine": 0.1, "feature_norm": 8.0}
    after = dict(before, mean_pairwise_cosine=0.99)
    assert verdict(before, after)[0] == "COLLAPSED"


def test_healthy_change_passes():
    """Adaptation should change the representation without destroying it."""
    before = {"mean_variance": 1.0, "effective_rank": 50.0, "dimensions": 64,
              "rank_fraction": 0.78, "mean_pairwise_cosine": 0.05, "feature_norm": 8.0}
    after = {"mean_variance": 0.8, "effective_rank": 42.0, "dimensions": 64,
             "rank_fraction": 0.66, "mean_pairwise_cosine": 0.12, "feature_norm": 7.2}
    status, problems = verdict(before, after)
    assert status == "healthy"
    assert problems == []


def test_dimensions_reported():
    assert feature_statistics(torch.randn(100, 1024))["dimensions"] == 1024