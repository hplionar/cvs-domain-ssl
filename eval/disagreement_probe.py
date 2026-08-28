#!/usr/bin/env python3
"""Is annotator disagreement predictable from the image?

Stratified evaluation showed that between-arm differences on SAGES live almost
entirely on frames where all three raters agreed: AP lift over chance spans
0.079-0.440 on unanimous frames and 0.095-0.166 on contested ones. Whether that
flat contested column can be moved at all depends on a prior question -- does
the image carry information about *whether raters will disagree*, or is
disagreement a property of the raters rather than the frame?

The experiment reuses the cached frozen features and changes only the target.
For each criterion the label is 1 when the vote count is 1 or 2 (a rater
dissented) and 0 when it is 0 or 3 (unanimous).

Three controls, without which the headline number is uninterpretable:

1. **A reference task.** The identical classifier is also fitted to the CVS
   consensus label. A disagreement AUC of 0.68 means something different if the
   same features reach 0.82 on the criterion than if they reach 0.70.

2. **The consensus-label baseline.** Contested frames are far more positive
   than unanimous ones (C1: 0.31 against 0.05 on val), so anything correlated
   with the criterion is automatically correlated with disagreement. The AUC of
   the consensus label used *alone* as a predictor of disagreement is the floor
   the visual features must clear.

3. **Within-label AUC.** Disagreement predicted separately among
   consensus-positive and consensus-negative frames. This removes the
   correlation in (2) by construction and is the number to trust.

Protocol notes. Regularisation is selected by GroupKFold over video_id inside
the training split -- frames from one procedure share patient, anatomy, camera
and illumination, so an ungrouped fold would leak. The validation split is
scored once, after selection. Features are mean-pooled patch tokens, matching
the mean head used in the main probe.

Usage:
    python eval/disagreement_probe.py \\
        --manifest metadata/sages_frames_internal_split.csv \\
        --train-cache ../cache/dinov3_b/sages/train \\
        --val-cache   ../cache/dinov3_b/sages/val \\
        --arm dinov3_b \\
        --output-dir ../outputs/disagreement_probe
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

CRITERIA = ("c1", "c2", "c3")
C_GRID = (0.001, 0.01, 0.1, 1.0)

#: Bytes of one decoded block while pooling. Peak is roughly 1.5x this, since
#: the fp16 block and its fp32 copy are briefly alive together.
POOL_BLOCK_BYTES = 1 << 30


def load_pooled(cache_dir: Path) -> tuple[np.ndarray, list[str]]:
    """Mean-pooled patch tokens plus the sample_id of each row, in cache order."""
    cache_dir = Path(cache_dir)
    tokens = np.load(cache_dir / "tokens.npy", mmap_mode="r")
    n, n_tokens, dim = tokens.shape

    block = max(1, int(POOL_BLOCK_BYTES // max(n_tokens * dim * 4, 1)))
    pooled = np.empty((n, dim), dtype=np.float32)
    for start in range(0, n, block):
        stop = min(start + block, n)
        pooled[start:stop] = np.asarray(tokens[start:stop], dtype=np.float32).mean(axis=1)

    with open(cache_dir / "index.csv", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if [int(r["row"]) for r in rows] != list(range(len(rows))):
        raise ValueError(f"{cache_dir/'index.csv'} rows are not 0..N-1 in order.")
    if len(rows) != n:
        raise ValueError(f"{cache_dir}: index.csv has {len(rows)} rows, tokens.npy has {n}.")
    return pooled, [r["sample_id"] for r in rows]


def align(meta: pd.DataFrame, sample_ids: list[str]) -> pd.DataFrame:
    indexed = meta.set_index("sample_id")
    unknown = [s for s in sample_ids if s not in indexed.index]
    if unknown:
        raise ValueError(f"{len(unknown)} sample_ids absent from manifest, e.g. {unknown[:3]}")
    return indexed.loc[sample_ids].reset_index()


def disagreement_target(df: pd.DataFrame, criterion: str) -> np.ndarray:
    votes = df[[f"{criterion}_rater{r}" for r in (1, 2, 3)]].to_numpy(dtype=int).sum(axis=1)
    return ((votes == 1) | (votes == 2)).astype(int)


def fit_and_score(
    x_train: np.ndarray,
    y_train: np.ndarray,
    groups: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple[np.ndarray, float, dict[str, float]]:
    """Select C by grouped CV inside train, then score the validation split once."""
    if np.unique(y_train).size < 2:
        raise ValueError("Training target has one class.")

    scaler = StandardScaler().fit(x_train)
    xt, xv = scaler.transform(x_train), scaler.transform(x_val)

    n_splits = min(5, len(np.unique(groups)))
    folds = list(GroupKFold(n_splits=n_splits).split(xt, y_train, groups))

    cv_scores = {}
    for c in C_GRID:
        aucs = []
        for tr, te in folds:
            if np.unique(y_train[tr]).size < 2 or np.unique(y_train[te]).size < 2:
                continue
            model = LogisticRegression(C=c, max_iter=2000, class_weight="balanced")
            model.fit(xt[tr], y_train[tr])
            aucs.append(roc_auc_score(y_train[te], model.predict_proba(xt[te])[:, 1]))
        cv_scores[c] = float(np.mean(aucs)) if aucs else float("nan")

    best_c = max(cv_scores, key=lambda k: (cv_scores[k] if not np.isnan(cv_scores[k]) else -1))
    final = LogisticRegression(C=best_c, max_iter=2000, class_weight="balanced")
    final.fit(xt, y_train)
    scores = final.predict_proba(xv)[:, 1]
    auc = float(roc_auc_score(y_val, scores)) if np.unique(y_val).size > 1 else float("nan")
    return scores, auc, {"selected_C": best_c, "cv_auc": cv_scores}


def binary_predictor_auc(feature: np.ndarray, target: np.ndarray) -> float:
    """AUC of a single binary variable used as a score. The floor to beat."""
    if np.unique(target).size < 2 or np.unique(feature).size < 2:
        return float("nan")
    return float(roc_auc_score(target, feature))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True)
    p.add_argument("--train-cache", required=True)
    p.add_argument("--val-cache", required=True)
    p.add_argument("--arm", required=True, help="label for the output file")
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    meta = pd.read_csv(args.manifest)

    print(f"loading {args.train_cache}")
    x_train, ids_train = load_pooled(Path(args.train_cache))
    print(f"loading {args.val_cache}")
    x_val, ids_val = load_pooled(Path(args.val_cache))

    tr = align(meta, ids_train)
    va = align(meta, ids_val)
    groups = tr["video_id"].to_numpy()
    print(f"train {x_train.shape}  val {x_val.shape}  "
          f"{tr.video_id.nunique()} / {va.video_id.nunique()} videos\n")

    results: dict[str, Any] = {"arm": args.arm, "criteria": {}}

    header = (f"{'crit':<5}{'prev':>7}{'DISAGREE':>10}{'label-only':>12}"
              f"{'among pos':>11}{'among neg':>11}{'CVS ref':>10}")
    print(header)
    print("-" * len(header))

    for criterion in CRITERIA:
        d_train = disagreement_target(tr, criterion)
        d_val = disagreement_target(va, criterion)
        y_train = tr[f"{criterion}_consensus"].to_numpy(dtype=int)
        y_val = va[f"{criterion}_consensus"].to_numpy(dtype=int)

        # Main question: predict disagreement from the frozen features.
        scores, auc_dis, info_dis = fit_and_score(x_train, d_train, groups, x_val, d_val)

        # Control 1: the same features on the criterion itself, for scale.
        _, auc_cvs, info_cvs = fit_and_score(x_train, y_train, groups, x_val, y_val)

        # Control 2: the consensus label alone as a predictor of disagreement.
        auc_label = binary_predictor_auc(y_val.astype(float), d_val)

        # Control 3: within-label AUC, which removes that correlation entirely.
        within = {}
        for name, mask in (("pos", y_val == 1), ("neg", y_val == 0)):
            if mask.sum() > 10 and np.unique(d_val[mask]).size > 1:
                within[name] = float(roc_auc_score(d_val[mask], scores[mask]))
            else:
                within[name] = float("nan")

        print(f"{criterion:<5}{d_val.mean():>7.3f}{auc_dis:>10.4f}{auc_label:>12.4f}"
              f"{within['pos']:>11.4f}{within['neg']:>11.4f}{auc_cvs:>10.4f}")

        results["criteria"][criterion] = {
            "disagreement_prevalence_val": float(d_val.mean()),
            "auc_disagreement": auc_dis,
            "auc_disagreement_within_positive": within["pos"],
            "auc_disagreement_within_negative": within["neg"],
            "auc_consensus_label_alone": auc_label,
            "auc_cvs_reference": auc_cvs,
            "n_val_pos": int((y_val == 1).sum()),
            "n_val_neg": int((y_val == 0).sum()),
            "selection_disagreement": info_dis,
            "selection_cvs": info_cvs,
        }

    print("\nDISAGREE   : predicting whether a rater dissented, from the frozen features")
    print("label-only : consensus label alone as the predictor -- the floor to beat")
    print("among pos/neg : the same, within one consensus class, so the correlation")
    print("             with the criterion cannot contribute. Trust these.")
    print("CVS ref    : the same classifier on the criterion itself, for scale")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"disagreement_{args.arm}.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwritten to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
