"""Tests for scripts/plot_curves.py."""

from __future__ import annotations

import json

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")

from scripts.plot_curves import (  # noqa: E402
    apply_style,
    load_histories,
    load_learning_curve,
    plot_learning,
    plot_training,
    stack_metric,
)


def write_run(tmp_path, name, *, seeds=3, epochs=10, ragged=False, curve=True, offset=0.0):
    run = tmp_path / name
    run.mkdir(parents=True, exist_ok=True)
    for seed in range(seeds):
        n = epochs - (seed if ragged else 0)
        rows = [
            {
                "epoch": e,
                "train_loss": 1.0 / e,
                "mAP": min(0.95, 0.3 + 0.05 * e + 0.01 * seed + offset),
                "mean_auc": 0.5 + 0.02 * e,
                "mean_bacc": 0.5 + 0.01 * e,
            }
            for e in range(1, n + 1)
        ]
        (run / f"history_seed{seed}.json").write_text(json.dumps(rows))

    if curve:
        points = []
        for fraction in (0.1, 0.25, 0.5, 1.0):
            maps = [0.3 + 0.2 * fraction + offset + 0.01 * s for s in range(seeds)]
            points.append({
                "fraction": fraction,
                "mean_map": float(np.mean(maps)),
                "std_map": float(np.std(maps, ddof=1)),
                "mean_videos": 120 * fraction,
                "mean_samples": 6960 * fraction,
                "per_seed": [],
            })
        (run / "learning_curve.json").write_text(json.dumps(points))
    return run


# -- loading --------------------------------------------------------------


def test_load_histories(tmp_path):
    assert sorted(load_histories(write_run(tmp_path, "r"))) == [0, 1, 2]


def test_missing_history_raises(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="history_seed"):
        load_histories(tmp_path / "empty")


def test_missing_learning_curve_names_the_flag(tmp_path):
    run = write_run(tmp_path, "r", curve=False)
    with pytest.raises(FileNotFoundError, match="--label-fractions"):
        load_learning_curve(run)


# -- aggregation ----------------------------------------------------------


def test_stack_metric_returns_mean_and_spread(tmp_path):
    epochs, mean, std = stack_metric(load_histories(write_run(tmp_path, "r")), "mAP")
    assert len(epochs) == len(mean) == len(std) == 10
    assert np.all(std >= 0)


def test_ragged_seeds_truncate_to_shortest(tmp_path):
    """Early stopping ends runs at different epochs. A ragged mean would be
    computed over a shrinking number of seeds and drift for that reason alone."""
    histories = load_histories(write_run(tmp_path, "r", seeds=3, epochs=10, ragged=True))
    lengths = {len(rows) for rows in histories.values()}
    assert lengths == {10, 9, 8}
    epochs, mean, _ = stack_metric(histories, "mAP")
    assert len(epochs) == 8 == len(mean)


def test_single_seed_gives_zero_spread(tmp_path):
    _, _, std = stack_metric(load_histories(write_run(tmp_path, "r", seeds=1)), "mAP")
    assert np.all(std == 0)


# -- rendering ------------------------------------------------------------


def test_training_plot_written(tmp_path):
    apply_style()
    run = write_run(tmp_path, "r")
    out = tmp_path / "training.pdf"
    plot_training([run], ["run"], ["mAP", "train_loss"], out)
    assert out.is_file() and out.stat().st_size > 1000


def test_training_plot_handles_multiple_runs(tmp_path):
    apply_style()
    runs = [write_run(tmp_path, "a"), write_run(tmp_path, "b", offset=0.05)]
    out = tmp_path / "cmp.pdf"
    plot_training(runs, ["A", "B"], ["mAP"], out)
    assert out.is_file()


def test_learning_plot_written(tmp_path):
    apply_style()
    out = tmp_path / "learning.pdf"
    plot_learning([write_run(tmp_path, "r")], ["run"], out)
    assert out.is_file() and out.stat().st_size > 1000


def test_learning_plot_two_arms(tmp_path):
    """The two-arm case is the dissertation figure: baseline against adapted."""
    apply_style()
    runs = [write_run(tmp_path, "base"), write_run(tmp_path, "adapted", offset=0.06)]
    out = tmp_path / "arms.pdf"
    plot_learning(runs, ["baseline", "adapted"], out)
    assert out.is_file()


def test_learning_plot_video_axis(tmp_path):
    apply_style()
    out = tmp_path / "videos.pdf"
    plot_learning([write_run(tmp_path, "r")], ["run"], out, x_axis="videos")
    assert out.is_file()


def test_output_is_vector_pdf(tmp_path):
    """PDF keeps the figure vector for LaTeX inclusion."""
    apply_style()
    out = tmp_path / "vector.pdf"
    plot_learning([write_run(tmp_path, "r")], ["run"], out)
    assert out.read_bytes()[:4] == b"%PDF"