"""Tests for scripts/analyse_corpus.py.

Fleiss' kappa is checked against hand-computable cases, since a silently wrong
agreement statistic would be reported in the dissertation without any obvious
symptom.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from scripts.analyse_corpus import (
    _entropy,
    analyse_agreement,
    effective_sample_size,
    fleiss_kappa,
    interpret_kappa,
)


# -- Fleiss' kappa --------------------------------------------------------


def test_perfect_agreement_with_mixed_prevalence():
    ratings = np.array([[1, 1, 1]] * 50 + [[0, 0, 0]] * 50)
    assert fleiss_kappa(ratings) == pytest.approx(1.0)


def test_perfect_agreement_on_one_category_is_undefined():
    """All raters always saying 'no' is not evidence of agreement: chance
    agreement is also 1, so kappa is 0/0."""
    assert np.isnan(fleiss_kappa(np.zeros((50, 3), dtype=int)))


def test_maximal_disagreement_is_negative():
    ratings = np.array([[1, 0, 0], [0, 1, 1]] * 25)
    assert fleiss_kappa(ratings) < 0


def test_kappa_lies_in_range():
    rng = np.random.default_rng(0)
    for p in (0.1, 0.3, 0.5):
        ratings = (rng.random((200, 3)) < p).astype(int)
        k = fleiss_kappa(ratings)
        assert -1.0 <= k <= 1.0


def test_independent_raters_give_kappa_near_zero():
    """Kappa corrects for chance, so independent raters should score near 0
    even at low prevalence where raw agreement is high."""
    rng = np.random.default_rng(1)
    ratings = (rng.random((5000, 3)) < 0.12).astype(int)
    assert abs(fleiss_kappa(ratings)) < 0.05


def test_correlated_raters_beat_independent_ones():
    rng = np.random.default_rng(2)
    truth = (rng.random(2000) < 0.2).astype(int)
    noisy = np.stack([
        np.where(rng.random(2000) < 0.1, 1 - truth, truth) for _ in range(3)
    ], axis=1)
    independent = (rng.random((2000, 3)) < 0.2).astype(int)
    assert fleiss_kappa(noisy) > fleiss_kappa(independent) + 0.3


def test_too_few_raters_or_items():
    assert np.isnan(fleiss_kappa(np.zeros((0, 3))))
    assert np.isnan(fleiss_kappa(np.ones((10, 1))))


def test_interpretation_labels():
    assert interpret_kappa(0.1) == "slight"
    assert interpret_kappa(0.5) == "fair" or interpret_kappa(0.5) == "moderate"
    assert interpret_kappa(0.9) == "almost perfect"
    assert interpret_kappa(float("nan")) == "undefined"


# -- entropy --------------------------------------------------------------


def test_entropy_of_uniform_distribution():
    assert _entropy([25, 25, 25, 25]) == pytest.approx(2.0)


def test_entropy_of_single_category_is_zero():
    assert _entropy([100]) == pytest.approx(0.0)


def test_entropy_increases_with_diversity():
    assert _entropy([90, 10]) < _entropy([50, 50])


# -- agreement over label directories -------------------------------------


def _write_labels(tmp_path, n_videos=5, n_frames=18, agree=True):
    rng = np.random.default_rng(0)
    dirs = []
    for v in range(n_videos):
        d = tmp_path / f"video_{v:03d}"
        d.mkdir(parents=True)
        truth = (rng.random(n_frames) < 0.3).astype(int)
        cols = {"frame_id": np.arange(n_frames) * 150}
        for c in ("c1", "c2", "c3"):
            for r in (1, 2, 3):
                if agree:
                    cols[f"{c}_rater{r}"] = truth
                else:
                    cols[f"{c}_rater{r}"] = (rng.random(n_frames) < 0.3).astype(int)
        pd.DataFrame(cols).to_csv(d / "frame.csv", index=False)
        pd.DataFrame([{
            **{f"{c}_rater{r}": 0 for c in ("c1", "c2", "c3") for r in (1, 2, 3)},
            "confidence_rater1": 0.75, "confidence_rater2": 1.0, "confidence_rater3": 0.5,
        }]).to_csv(d / "video.csv", index=False)
        dirs.append(d)
    return dirs


def test_agreement_over_directories(tmp_path):
    result = analyse_agreement(_write_labels(tmp_path, agree=True))
    assert set(result["criteria"]) == {"c1", "c2", "c3"}
    assert result["criteria"]["c1"]["fleiss_kappa"] == pytest.approx(1.0)
    assert result["criteria"]["c1"]["split_decision_fraction"] == 0.0


def test_disagreement_is_detected(tmp_path):
    result = analyse_agreement(_write_labels(tmp_path, n_videos=20, agree=False))
    assert result["criteria"]["c1"]["fleiss_kappa"] < 0.3
    assert result["criteria"]["c1"]["split_decision_fraction"] > 0.1


def test_rater_confidence_collected(tmp_path):
    result = analyse_agreement(_write_labels(tmp_path))
    conf = result["rater_confidence"]
    assert conf["n_videos"] == 5
    assert conf["mean"] == pytest.approx(0.75, abs=0.01)


def test_missing_columns_are_skipped(tmp_path):
    d = tmp_path / "sparse"
    d.mkdir()
    pd.DataFrame({"frame_id": [0, 150]}).to_csv(d / "frame.csv", index=False)
    assert analyse_agreement([d])["criteria"] == {}


# -- effective sample size ------------------------------------------------


def test_effective_sample_size_from_decorrelation():
    geometry = {"duration_s": {"mean": 90.0}, "fps": [30.0]}
    decorrelation = {"decorrelation_time_s": 3.0, "half_decay_time_s": 1.5}
    result = effective_sample_size(geometry, decorrelation, num_videos_total=560)

    assert result["effective_samples_per_video"] == pytest.approx(30.0)
    assert result["frames_per_video"] == 2700
    assert result["redundancy_factor"] == pytest.approx(90.0)
    assert result["corpus_frames"] == 2700 * 560
    assert result["corpus_effective_samples"] == 30 * 560
    assert result["basis"] == "decorrelation_time"


def test_falls_back_to_half_decay():
    geometry = {"duration_s": {"mean": 90.0}, "fps": [30.0]}
    decorrelation = {"decorrelation_time_s": None, "half_decay_time_s": 2.0}
    result = effective_sample_size(geometry, decorrelation, 560)
    assert result["basis"] == "half_decay"
    assert result["effective_samples_per_video"] == pytest.approx(45.0)


def test_missing_inputs_reported():
    assert "error" in effective_sample_size({}, {}, 100)


def test_effective_size_is_far_below_frame_count():
    """The point of the measurement: frames overstate corpus size."""
    geometry = {"duration_s": {"mean": 90.0}, "fps": [30.0]}
    result = effective_sample_size(geometry, {"decorrelation_time_s": 3.0}, 560)
    assert result["corpus_effective_samples"] < result["corpus_frames"] / 50