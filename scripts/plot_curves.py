#!/usr/bin/env python3
"""Plot training and learning curves from probe run directories.

Two distinct things are both called learning curves; this script produces both
and keeps them separate.

**Training curves** plot a metric against epoch. They are monitoring: used to
see whether a head is converging or oscillating, and whether fine-tuning has
begun to overfit.

**Learning curves** plot performance against the quantity of labelled training
data. These are a dissertation figure rather than monitoring. If domain-adaptive
pretraining works, its advantage should be largest in the low-label regime,
because that is precisely where a better representation substitutes for absent
supervision. A curve showing the adapted encoder above the baseline at 10% of
labels and converging at 100% is a stronger claim than a single mAP difference.

Reads ``history_seed*.json`` and ``learning_curve.json`` written by
``train/train_probe_cached.py``. Nothing here reads TensorBoard event files:
``history.json`` is the authoritative record, being deterministic, diffable and
version-controlled.

Usage:
    python scripts/plot_curves.py --mode training --run-dir outputs/probe/mae_b
    python scripts/plot_curves.py --mode learning \\
        --run-dir outputs/probe/videomae_baseline outputs/probe/videomae_adapted \\
        --labels "VideoMAE (original)" "VideoMAE (surgical SSL)" \\
        --out figures/learning_curve.pdf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


# Okabe-Ito: distinguishable under the common forms of colour vision deficiency
# and in greyscale print, which a bound dissertation may well be.
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
MARKERS = ["o", "s", "^", "D", "v", "P"]


def apply_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 8.5,
        "legend.frameon": False,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.4,
        "lines.markersize": 4,
    })


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------


def load_histories(run_dir: Path) -> dict[int, list[dict[str, Any]]]:
    histories = {}
    for path in sorted(run_dir.glob("history_seed*.json")):
        seed = int(path.stem.replace("history_seed", ""))
        histories[seed] = json.loads(path.read_text())
    if not histories:
        raise FileNotFoundError(f"No history_seed*.json in {run_dir}.")
    return histories


def load_learning_curve(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "learning_curve.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"No learning_curve.json in {run_dir}. Rerun train_probe_cached.py "
            f"with --label-fractions."
        )
    return json.loads(path.read_text())


def stack_metric(histories: dict[int, list[dict[str, Any]]], metric: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align seeds on epoch, truncating to the shortest.

    Early stopping ends runs at different epochs, so a ragged mean would be
    computed over a shrinking number of seeds and would drift for that reason
    alone.
    """
    series = []
    for rows in histories.values():
        values = [row[metric] for row in rows if metric in row]
        series.append(np.asarray(values, dtype=float))
    length = min(len(s) for s in series)
    matrix = np.stack([s[:length] for s in series])
    epochs = np.arange(1, length + 1)
    # ddof=1 is undefined for a single seed and emits a divide-by-zero warning;
    # warnings that are routinely ignored are how real ones get missed.
    spread = (
        np.nanstd(matrix, axis=0, ddof=1)
        if matrix.shape[0] > 1
        else np.zeros(length, dtype=float)
    )
    return epochs, np.nanmean(matrix, axis=0), spread


# --------------------------------------------------------------------------
# plots
# --------------------------------------------------------------------------


def plot_training(run_dirs: list[Path], labels: list[str], metrics: list[str], out: Path) -> None:
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.2 * len(metrics), 3.2), squeeze=False)

    for column, metric in enumerate(metrics):
        ax = axes[0][column]
        for index, (run_dir, label) in enumerate(zip(run_dirs, labels)):
            histories = load_histories(run_dir)
            epochs, mean, std = stack_metric(histories, metric)
            colour = PALETTE[index % len(PALETTE)]
            ax.plot(epochs, mean, color=colour, label=label)
            if np.any(std > 0):
                ax.fill_between(epochs, mean - std, mean + std, color=colour, alpha=0.18, linewidth=0)

            if metric == "mAP":
                peak = int(np.argmax(mean))
                ax.axvline(epochs[peak], color=colour, linestyle=":", linewidth=0.9, alpha=0.7)

        ax.set_xlabel("Epoch")
        ax.set_ylabel(_axis_label(metric))
        if column == 0 and len(run_dirs) > 1:
            ax.legend()

    fig.suptitle("Training curves (mean $\\pm$ s.d. across seeds)", y=1.02, fontsize=10)
    fig.savefig(out)
    plt.close(fig)


def plot_learning(run_dirs: list[Path], labels: list[str], out: Path, *, x_axis: str = "fraction") -> None:
    fig, ax = plt.subplots(figsize=(4.6, 3.4))

    curves = []
    for index, (run_dir, label) in enumerate(zip(run_dirs, labels)):
        curve = load_learning_curve(run_dir)
        curves.append(curve)

        if x_axis == "videos":
            x = np.array([point["mean_videos"] for point in curve])
        else:
            x = np.array([point["fraction"] * 100 for point in curve])
        mean = np.array([point["mean_map"] for point in curve])
        std = np.array([point["std_map"] for point in curve])

        colour = PALETTE[index % len(PALETTE)]
        ax.errorbar(
            x, mean, yerr=std, color=colour, marker=MARKERS[index % len(MARKERS)],
            capsize=2.5, elinewidth=0.9, label=label,
        )

    ax.set_xlabel("Training videos" if x_axis == "videos" else "Labelled training videos (%)")
    ax.set_ylabel("Validation mAP")
    ax.legend()

    # Where two arms are compared, the gap is the quantity of interest, so it is
    # annotated at the leftmost point rather than left to the reader's eye.
    if len(curves) == 2:
        low_a, low_b = curves[0][0]["mean_map"], curves[1][0]["mean_map"]
        gap = low_b - low_a
        x0 = curves[0][0]["fraction"] * 100 if x_axis != "videos" else curves[0][0]["mean_videos"]
        ax.annotate(
            f"$\\Delta$ = {gap:+.3f}",
            xy=(x0, (low_a + low_b) / 2),
            xytext=(8, 0), textcoords="offset points",
            fontsize=8, va="center", color="0.3",
        )

    fig.savefig(out)
    plt.close(fig)


def _axis_label(metric: str) -> str:
    return {
        "mAP": "Validation mAP",
        "mean_auc": "Validation mean ROC-AUC",
        "mean_bacc": "Validation mean balanced accuracy",
        "train_loss": "Training loss",
        "c1_ap": "C1 average precision",
        "c2_ap": "C2 average precision",
        "c3_ap": "C3 average precision",
    }.get(metric, metric)


# --------------------------------------------------------------------------


def main() -> int:
    args = parse_args()
    apply_style()

    run_dirs = [Path(d) for d in args.run_dir]
    labels = args.labels or [d.name for d in run_dirs]
    if len(labels) != len(run_dirs):
        raise SystemExit(f"Got {len(run_dirs)} run dirs but {len(labels)} labels.")

    out = Path(args.out) if args.out else run_dirs[0] / f"{args.mode}_curves.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "training":
        plot_training(run_dirs, labels, args.metrics, out)
    else:
        plot_learning(run_dirs, labels, out, x_axis=args.x_axis)

    print(f"Written to {out}")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["training", "learning"], default="training")
    p.add_argument("--run-dir", nargs="+", required=True)
    p.add_argument("--labels", nargs="*", default=None)
    p.add_argument("--metrics", nargs="*", default=["mAP", "train_loss"],
                   help="training mode only")
    p.add_argument("--x-axis", choices=["fraction", "videos"], default="fraction",
                   help="learning mode only")
    p.add_argument("--out", default=None, help="output path; .pdf keeps it vector for LaTeX")
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())