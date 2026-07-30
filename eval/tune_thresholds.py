#!/usr/bin/env python3
"""Tune per-criterion decision thresholds on validation, then apply to test.

Motivation (audit finding F3). ``metadata/experiments/baseline_test_evaluation_summary.md``
concludes that weighted BCE "confirms that class imbalance is important",
on the basis of balanced accuracy rising from 0.5293 to 0.6243 on Endoscapes.
But the threshold-free metrics did not improve:

    Exp001 -> Exp002   mAP 0.4142 -> 0.4066   AUC 0.7030 -> 0.6977
    Exp003 -> Exp004   mAP 0.4141 -> 0.4220   AUC 0.7252 -> 0.7323

Balanced accuracy was computed at a fixed 0.5 cutoff. ``pos_weight`` shifts a
model's calibration, which moves its operating point relative to that cutoff.
The ranking quality of the representation is essentially unchanged; only where
the fixed threshold happens to fall has moved. The conclusion as written does
not follow.

The defensible comparison tunes the threshold per criterion on validation for
*both* losses, then reports balanced accuracy at the tuned threshold. If the
weighted model still wins, class imbalance handling has done something beyond
recalibration. If it does not, the original claim was an artefact.

Thresholds are always selected on validation and applied unchanged to test.
Selecting on test would be selecting on the quantity being reported.

Usage:
    python eval/tune_thresholds.py --run-dir outputs/probe/mae_b_endoscapes
    python eval/tune_thresholds.py --run-dir A --compare-run-dir B
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import balanced_accuracy_score

from eval.metrics import CRITERIA, sigmoid


def tune_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    num_candidates: int = 199,
) -> tuple[float, float]:
    """Return the threshold maximising balanced accuracy, and that value.

    Candidates are quantiles of the predicted probabilities rather than an even
    grid on [0, 1]: a poorly calibrated model may place all of its mass in a
    narrow band, where an even grid would have almost no resolution.
    """
    if len(np.unique(y_true)) < 2:
        return 0.5, float("nan")

    quantiles = np.linspace(0.005, 0.995, num_candidates)
    candidates = np.unique(np.quantile(y_prob, quantiles))

    best_threshold, best_score = 0.5, -1.0
    for threshold in candidates:
        score = balanced_accuracy_score(y_true, (y_prob >= threshold).astype(int))
        if score > best_score:
            best_threshold, best_score = float(threshold), float(score)
    return best_threshold, best_score


def tune_all_criteria(y_true: np.ndarray, logits: np.ndarray) -> dict[str, Any]:
    """Tune each CVS criterion independently.

    Independently because the criteria are separate binary decisions with
    different prevalences; a shared threshold would impose an arbitrary coupling.
    """
    probabilities = sigmoid(logits)
    thresholds, tuned, fixed = {}, {}, {}

    for index, name in enumerate(CRITERIA):
        threshold, score = tune_threshold(y_true[:, index], probabilities[:, index])
        thresholds[name] = threshold
        tuned[name] = score
        fixed[name] = (
            float(balanced_accuracy_score(y_true[:, index], (probabilities[:, index] >= 0.5).astype(int)))
            if len(np.unique(y_true[:, index])) > 1
            else float("nan")
        )

    return {
        "thresholds": thresholds,
        "bacc_tuned": tuned,
        "bacc_at_0.5": fixed,
        "mean_bacc_tuned": float(np.nanmean(list(tuned.values()))),
        "mean_bacc_at_0.5": float(np.nanmean(list(fixed.values()))),
    }


def apply_thresholds(
    y_true: np.ndarray, logits: np.ndarray, thresholds: dict[str, float]
) -> dict[str, Any]:
    """Evaluate with thresholds fixed in advance, i.e. selected elsewhere."""
    probabilities = sigmoid(logits)
    per_criterion = {}
    for index, name in enumerate(CRITERIA):
        if len(np.unique(y_true[:, index])) < 2:
            per_criterion[name] = float("nan")
            continue
        predictions = (probabilities[:, index] >= thresholds[name]).astype(int)
        per_criterion[name] = float(balanced_accuracy_score(y_true[:, index], predictions))
    return {
        "bacc": per_criterion,
        "mean_bacc": float(np.nanmean(list(per_criterion.values()))),
    }


def load_seed_logits(run_dir: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    payloads = {}
    for path in sorted(run_dir.glob("val_logits_seed*.npz")):
        seed = int(path.stem.replace("val_logits_seed", ""))
        data = np.load(path)
        payloads[seed] = (data["targets"], data["logits"])
    if not payloads:
        raise FileNotFoundError(
            f"No val_logits_seed*.npz in {run_dir}. Run train_probe_cached.py first."
        )
    return payloads


def summarise_run(run_dir: Path) -> dict[str, Any]:
    """Tune per seed, then aggregate.

    Per seed rather than on pooled logits: each seed produces a differently
    calibrated model, so a single pooled threshold would suit none of them.
    """
    per_seed = {}
    for seed, (targets, logits) in load_seed_logits(run_dir).items():
        per_seed[seed] = tune_all_criteria(targets, logits)

    tuned = [entry["mean_bacc_tuned"] for entry in per_seed.values()]
    fixed = [entry["mean_bacc_at_0.5"] for entry in per_seed.values()]

    return {
        "run_dir": str(run_dir),
        "num_seeds": len(per_seed),
        "per_seed": per_seed,
        "mean_bacc_at_0.5": {"mean": float(np.mean(fixed)), "std": _std(fixed)},
        "mean_bacc_tuned": {"mean": float(np.mean(tuned)), "std": _std(tuned)},
        "gain_from_tuning": float(np.mean(tuned) - np.mean(fixed)),
    }


def _std(values: list[float]) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def _render(summary: dict[str, Any], label: str) -> None:
    print(f"\n{label}")
    print(f"  seeds                 {summary['num_seeds']}")
    print(f"  mean BAcc @ 0.5       {summary['mean_bacc_at_0.5']['mean']:.4f} "
          f"+/- {summary['mean_bacc_at_0.5']['std']:.4f}")
    print(f"  mean BAcc tuned       {summary['mean_bacc_tuned']['mean']:.4f} "
          f"+/- {summary['mean_bacc_tuned']['std']:.4f}")
    print(f"  gain from tuning      {summary['gain_from_tuning']:+.4f}")

    first = next(iter(summary["per_seed"].values()))
    thresholds = ", ".join(f"{k}={v:.3f}" for k, v in first["thresholds"].items())
    print(f"  thresholds (seed 0)   {thresholds}")


def main() -> int:
    args = parse_args()

    primary = summarise_run(Path(args.run_dir))
    _render(primary, f"Run A: {args.run_dir}")

    output: dict[str, Any] = {"primary": primary}

    if args.compare_run_dir:
        secondary = summarise_run(Path(args.compare_run_dir))
        _render(secondary, f"Run B: {args.compare_run_dir}")

        fixed_delta = (
            secondary["mean_bacc_at_0.5"]["mean"] - primary["mean_bacc_at_0.5"]["mean"]
        )
        tuned_delta = (
            secondary["mean_bacc_tuned"]["mean"] - primary["mean_bacc_tuned"]["mean"]
        )
        output["comparison"] = {
            "delta_at_0.5": fixed_delta,
            "delta_tuned": tuned_delta,
            "artefact_fraction": (
                float(1.0 - tuned_delta / fixed_delta) if abs(fixed_delta) > 1e-9 else None
            ),
        }

        print("\nB minus A")
        print(f"  at fixed 0.5          {fixed_delta:+.4f}")
        print(f"  at tuned thresholds   {tuned_delta:+.4f}")
        if abs(fixed_delta) > 1e-9:
            share = 1.0 - tuned_delta / fixed_delta
            print(f"  attributable to recalibration: {share:.0%}")
            if share > 0.5:
                print("\n  More than half of the apparent difference disappears once "
                      "both models are evaluated at their own tuned operating point. "
                      "It reflects calibration, not representation quality.")

    out_path = Path(args.output or Path(args.run_dir) / "thresholds.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2)
    print(f"\nWritten to {out_path}")

    print("\nNote: thresholds selected on validation are optimistic when reported "
          "on that same split. Apply them unchanged to test for the reported figure.")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True, help="probe output directory")
    p.add_argument("--compare-run-dir", default=None,
                   help="second run, e.g. the weighted-BCE counterpart")
    p.add_argument("--output", default=None)
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())