#!/usr/bin/env python3
"""Annotation-ceiling and agreement-stratified evaluation for SAGES CVS.

Every mAP reported on SAGES is scored against the majority vote of three
raters, as though that vote were ground truth. Fleiss' kappa on this manifest
is 0.47-0.54, so it is not. This script quantifies two consequences.

**The ceiling.** Leave-one-rater-out: rater *i*'s label is treated as a
prediction and the majority of the other two as the reference, rotating over
all three and averaging. Frames where the other two split 1-1 have no majority
and are dropped; their count is reported. Rater labels are binary, so a
precision-recall curve built from them is degenerate and AP is meaningless --
the ceiling is reported in balanced accuracy and F1, and models are compared
against it in the same units.

**The strata.** Frames are split into unanimous (3-0) and contested (2-1), and
every arm is scored on each separately.

Two traps this script avoids:

1. AP depends on prevalence, and the contested stratum is far more positive
   than the unanimous one. Comparing raw AP across strata measures the
   prevalence difference. AUC is reported as the primary stratified metric
   because it is prevalence-invariant; AP is reported alongside its own random
   baseline (which equals the prevalence) so the lift is visible.

2. Comparing a model to the human ceiling needs a threshold. Tuning one on the
   evaluation set would inflate the model side of the comparison, so the
   threshold is set by prevalence matching -- the model is made to predict
   positive at the same rate the label occurs. This uses one number from the
   evaluation labels and no search.

Row alignment: `evaluate()` in train_probe_cached.py iterates the validation
loader with shuffle=False, and CachedFeatures preserves index.csv order, so
row *i* of a val_logits_seed*.npz is row *i* of the cache's index.csv. The
script asserts this by checking lengths and by re-deriving the targets.

Usage
-----
Ceiling only, no model files needed:

    python eval/rater_agreement.py \
        --manifest metadata/sages_frames_internal_split.csv \
        --split val --output-dir outputs/rater_agreement

Add arms (repeatable; name=probe_dir:cache_dir):

    python eval/rater_agreement.py \
        --manifest metadata/sages_frames_internal_split.csv \
        --split val --output-dir outputs/rater_agreement \
        --arm dinov2_b=outputs/cvs-domain-ssl/probe/dinov2_b_sages_mean:cache/dinov2_b/sages/val \
        --arm dinov3_b=outputs/cvs-domain-ssl/probe/dinov3_b_sages_mean:cache/dinov3_b/sages/val
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)

CRITERIA = ("c1", "c2", "c3")
RATERS = (1, 2, 3)


# --------------------------------------------------------------------------
# ceiling
# --------------------------------------------------------------------------


def rater_matrix(df: pd.DataFrame, criterion: str) -> np.ndarray:
    """[N, 3] binary matrix of the three raters' votes for one criterion."""
    cols = [f"{criterion}_rater{r}" for r in RATERS]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Manifest is missing per-rater columns: {missing}")
    return df[cols].to_numpy(dtype=int)


def leave_one_rater_out(df: pd.DataFrame, criterion: str) -> dict[str, Any]:
    """Score each rater against the majority of the other two.

    Ties (the other two split 1-1) admit no majority and are excluded rather
    than broken by a rule, since any rule would be an invented label.
    """
    votes = rater_matrix(df, criterion)
    per_rater, dropped = [], 0

    for held_out in range(3):
        others = [j for j in range(3) if j != held_out]
        pair = votes[:, others]
        agreed = pair[:, 0] == pair[:, 1]
        dropped += int((~agreed).sum())

        reference = pair[agreed, 0]
        prediction = votes[agreed, held_out]
        if np.unique(reference).size < 2:
            continue

        per_rater.append(
            {
                "rater": held_out + 1,
                "n": int(agreed.sum()),
                "bacc": float(balanced_accuracy_score(reference, prediction)),
                "f1": float(f1_score(reference, prediction, zero_division=0)),
                "recall": float(
                    ((prediction == 1) & (reference == 1)).sum()
                    / max((reference == 1).sum(), 1)
                ),
                "specificity": float(
                    ((prediction == 0) & (reference == 0)).sum()
                    / max((reference == 0).sum(), 1)
                ),
            }
        )

    if not per_rater:
        return {"criterion": criterion, "usable": False}

    return {
        "criterion": criterion,
        "usable": True,
        "bacc": float(np.mean([r["bacc"] for r in per_rater])),
        "f1": float(np.mean([r["f1"] for r in per_rater])),
        "recall": float(np.mean([r["recall"] for r in per_rater])),
        "specificity": float(np.mean([r["specificity"] for r in per_rater])),
        "ties_dropped_per_rater": dropped // 3,
        "per_rater": per_rater,
    }


def agreement_strata(df: pd.DataFrame) -> dict[str, Any]:
    """Unanimous/contested split sizes, prevalences, and positive mass."""
    out = {}
    for criterion in CRITERIA:
        votes = rater_matrix(df, criterion).sum(axis=1)
        label = df[f"{criterion}_consensus"].to_numpy(dtype=int).astype(bool)
        unanimous = (votes == 0) | (votes == 3)

        positives = int(label.sum())
        contested_positives = int((label & ~unanimous).sum())
        out[criterion] = {
            "n_unanimous": int(unanimous.sum()),
            "n_contested": int((~unanimous).sum()),
            "prevalence_unanimous": float(label[unanimous].mean()) if unanimous.any() else float("nan"),
            "prevalence_contested": float(label[~unanimous].mean()) if (~unanimous).any() else float("nan"),
            "n_positives": positives,
            "positives_contested": contested_positives,
            # The share of the positive class -- which is what AP is computed
            # over -- that sits on frames a rater dissented from.
            "share_positives_contested": float(contested_positives / max(positives, 1)),
        }
    return out


# --------------------------------------------------------------------------
# model side
# --------------------------------------------------------------------------


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-x))


def read_cache_sample_ids(cache_dir: Path) -> list[str]:
    with open(cache_dir / "index.csv", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"{cache_dir/'index.csv'} is empty.")
    if [int(r["row"]) for r in rows] != list(range(len(rows))):
        raise ValueError(
            f"{cache_dir/'index.csv'} rows are not 0..N-1 in order; the join "
            f"assumption of this script does not hold for that cache."
        )
    return [r["sample_id"] for r in rows]


def prevalence_matched_threshold(scores: np.ndarray, prevalence: float) -> float:
    """Threshold making the predicted positive rate equal the label prevalence.

    Chosen over a tuned threshold because tuning on the evaluation split would
    put the model and the human on different footings: the raters got no such
    calibration.
    """
    if not 0.0 < prevalence < 1.0:
        return float("inf")
    return float(np.quantile(scores, 1.0 - prevalence))


def score_stratum(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float | None
) -> dict[str, float]:
    if np.unique(y_true).size < 2:
        return {k: float("nan") for k in ("ap", "auc", "bacc", "prevalence", "ap_lift", "n")}
    prevalence = float(y_true.mean())
    if threshold is None:
        threshold = prevalence_matched_threshold(y_score, prevalence)
    y_pred = (y_score >= threshold).astype(int)
    ap = float(average_precision_score(y_true, y_score))
    return {
        "n": int(y_true.size),
        "prevalence": prevalence,
        "ap": ap,
        # A random ranker scores AP == prevalence, so raw AP is not comparable
        # between strata of different prevalence. The lift is.
        "ap_lift": ap - prevalence,
        "auc": float(roc_auc_score(y_true, y_score)),
        "bacc": float(balanced_accuracy_score(y_true, y_pred)),
    }


def evaluate_arm(
    name: str, probe_dir: Path, cache_dir: Path, meta: pd.DataFrame,
    logits_prefix: str = "val_logits",
) -> dict[str, Any]:
    """Score one arm on the full split and on each agreement stratum.

    The prefix is a parameter because an arm scored against more than one test
    cache writes to distinct filenames, and reading the wrong one would compare
    the right arm on the wrong split. The row-count check below catches that
    when the splits differ in size, but not when they happen to match.
    """
    logit_files = sorted(probe_dir.glob(f"{logits_prefix}_seed*.npz"))
    if not logit_files:
        raise FileNotFoundError(f"No {logits_prefix}_seed*.npz in {probe_dir}")

    sample_ids = read_cache_sample_ids(cache_dir)
    meta_indexed = meta.set_index("sample_id")
    unknown = [s for s in sample_ids if s not in meta_indexed.index]
    if unknown:
        raise ValueError(
            f"{len(unknown)} sample_ids from {cache_dir} are absent from the "
            f"manifest, e.g. {unknown[:3]}"
        )
    aligned = meta_indexed.loc[sample_ids].reset_index()

    per_seed = []
    for path in logit_files:
        payload = np.load(path)
        logits, targets = payload["logits"], payload["targets"]
        if logits.shape[0] != len(sample_ids):
            raise ValueError(
                f"{path.name} has {logits.shape[0]} rows but {cache_dir} has "
                f"{len(sample_ids)}. These do not correspond."
            )
        # Independent check that the join is right: the targets saved with the
        # logits must equal the consensus labels the manifest gives for the
        # sample_ids in cache order.
        expected = aligned[[f"{c}_consensus" for c in CRITERIA]].to_numpy(dtype=float)
        if not np.allclose(targets, expected):
            raise ValueError(
                f"Targets in {path.name} disagree with the manifest consensus "
                f"labels under the assumed row order. The join is wrong."
            )

        probs = sigmoid(logits)
        seed_result: dict[str, Any] = {"file": path.name}
        for i, criterion in enumerate(CRITERIA):
            votes = rater_matrix(aligned, criterion).sum(axis=1)
            unanimous = (votes == 0) | (votes == 3)
            y = expected[:, i].astype(int)
            s = probs[:, i]

            # One threshold, fixed on the full split, reused for both strata --
            # a model does not know which stratum a frame belongs to.
            threshold = prevalence_matched_threshold(s, float(y.mean()))
            seed_result[criterion] = {
                "all": score_stratum(y, s, threshold),
                "unanimous": score_stratum(y[unanimous], s[unanimous], threshold),
                "contested": score_stratum(y[~unanimous], s[~unanimous], threshold),
            }
        per_seed.append(seed_result)

    summary: dict[str, Any] = {"arm": name, "n_seeds": len(per_seed)}
    for criterion in CRITERIA:
        summary[criterion] = {}
        for stratum in ("all", "unanimous", "contested"):
            for metric in ("ap", "ap_lift", "auc", "bacc", "prevalence"):
                values = [s[criterion][stratum][metric] for s in per_seed]
                summary[criterion].setdefault(stratum, {})[metric] = {
                    "mean": float(np.nanmean(values)),
                    "std": float(np.nanstd(values, ddof=1)) if len(values) > 1 else 0.0,
                }
    summary["per_seed"] = per_seed
    return summary


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------


def print_ceiling(ceiling: list[dict[str, Any]], strata: dict[str, Any], split: str, n: int) -> None:
    print(f"\nHuman ceiling -- leave-one-rater-out, split={split}, {n} frames")
    print(f"{'crit':<6}{'BAcc':>9}{'F1':>9}{'Recall':>9}{'Spec':>9}{'ties':>8}")
    usable = [c for c in ceiling if c["usable"]]
    for c in usable:
        print(f"{c['criterion']:<6}{c['bacc']:>9.4f}{c['f1']:>9.4f}"
              f"{c['recall']:>9.4f}{c['specificity']:>9.4f}{c['ties_dropped_per_rater']:>8d}")
    if usable:
        print(f"{'mean':<6}{np.mean([c['bacc'] for c in usable]):>9.4f}"
              f"{np.mean([c['f1'] for c in usable]):>9.4f}")

    print(f"\nAgreement strata")
    print(f"{'crit':<6}{'unan':>7}{'cont':>7}{'prev_u':>9}{'prev_c':>9}{'%pos contested':>16}")
    for criterion in CRITERIA:
        s = strata[criterion]
        print(f"{criterion:<6}{s['n_unanimous']:>7d}{s['n_contested']:>7d}"
              f"{s['prevalence_unanimous']:>9.4f}{s['prevalence_contested']:>9.4f}"
              f"{100*s['share_positives_contested']:>15.1f}%")


def print_arms(arms: list[dict[str, Any]], ceiling: list[dict[str, Any]]) -> None:
    if not arms:
        return
    ceiling_bacc = {c["criterion"]: c["bacc"] for c in ceiling if c["usable"]}

    print(f"\nAUC by agreement stratum (prevalence-invariant)")
    print(f"{'arm':<22}{'crit':<5}{'all':>9}{'unanimous':>12}{'contested':>12}{'delta':>9}")
    for arm in arms:
        for criterion in CRITERIA:
            a = arm[criterion]["all"]["auc"]["mean"]
            u = arm[criterion]["unanimous"]["auc"]["mean"]
            c = arm[criterion]["contested"]["auc"]["mean"]
            print(f"{arm['arm']:<22}{criterion:<5}{a:>9.4f}{u:>12.4f}{c:>12.4f}{u-c:>9.4f}")

    print(f"\nBalanced accuracy against the human ceiling (prevalence-matched threshold)")
    print(f"{'arm':<22}{'crit':<5}{'model':>9}{'human':>9}{'gap':>9}")
    for arm in arms:
        for criterion in CRITERIA:
            m = arm[criterion]["all"]["bacc"]["mean"]
            h = ceiling_bacc.get(criterion, float("nan"))
            print(f"{arm['arm']:<22}{criterion:<5}{m:>9.4f}{h:>9.4f}{m-h:>9.4f}")


# --------------------------------------------------------------------------


def parse_arm(spec: str) -> tuple[str, Path, Path]:
    if "=" not in spec or ":" not in spec.split("=", 1)[1]:
        raise argparse.ArgumentTypeError(
            f"--arm must be name=probe_dir:cache_dir, got {spec!r}"
        )
    name, rest = spec.split("=", 1)
    probe_dir, cache_dir = rest.rsplit(":", 1)
    return name, Path(probe_dir), Path(cache_dir)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True,
                   help="sages_frames_internal_split.csv, with per-rater columns")
    p.add_argument("--split", default="val", choices=["train", "val", "test", "all"])
    p.add_argument("--output-dir", required=True)
    p.add_argument("--arm", action="append", default=[], type=parse_arm,
                   help="name=probe_dir:cache_dir (repeatable)")
    p.add_argument("--split-column", default=None,
                   help="the manifest column holding the split. Defaults to "
                        "internal_split, which the official-test manifest does "
                        "not have; it uses split.")
    p.add_argument("--logits-prefix", default="test_logits",
                   help="filename stem of the saved logits. An arm scored "
                        "against more than one test cache writes to distinct "
                        "prefixes, and reading the wrong one would silently "
                        "compare the right arm on the wrong split.")
    args = p.parse_args()

    meta = pd.read_csv(args.manifest)
    column = args.split_column or (
        "internal_split" if "internal_split" in meta.columns else "split")
    if args.split != "all" and column not in meta.columns:
        raise ValueError(f"Manifest has no {column} column; pass --split-column.")
    split_df = meta if args.split == "all" else meta[meta[column] == args.split]
    if split_df.empty:
        raise ValueError(f"No rows for split={args.split}.")

    ceiling = [leave_one_rater_out(split_df, c) for c in CRITERIA]
    strata = agreement_strata(split_df)
    print_ceiling(ceiling, strata, args.split, len(split_df))

    arms = []
    for name, probe_dir, cache_dir in args.arm:
        try:
            arms.append(evaluate_arm(name, probe_dir, cache_dir, meta,
                                     logits_prefix=args.logits_prefix))
        except (FileNotFoundError, ValueError) as exc:
            print(f"\n[skipped] {name}: {exc}")
    print_arms(arms, ceiling)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "manifest": str(args.manifest),
        "split": args.split,
        "n_frames": int(len(split_df)),
        "n_videos": int(split_df.video_id.nunique()),
        "ceiling": ceiling,
        "strata": strata,
        "arms": arms,
    }
    with open(out_dir / f"rater_agreement_{args.split}.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nwritten to {out_dir/f'rater_agreement_{args.split}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
