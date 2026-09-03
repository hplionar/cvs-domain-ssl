#!/usr/bin/env python3
"""Annotation ceiling and agreement-stratified evaluation on Endoscapes.

The SAGES analyses in ``eval/rater_agreement.py`` and ``eval/bootstrap_strata.py``
read three per-rater columns. Endoscapes releases a **vote count** instead --
how many of its three annotators judged the criterion achieved -- so those
scripts cannot be pointed at it directly. This one works from the count.

**The count is sufficient.** With three exchangeable raters, holding one out
leaves the other two in a state the count determines:

    votes = 0   every held-out rater said no; the other two agree on no
    votes = 1   one rater said yes against a no majority; the other two
                held-out cases each face a split pair and are discarded
    votes = 2   two of three held-out cases face a split pair; the third is a
                no-voter against a yes majority
    votes = 3   every held-out rater said yes; the other two agree on yes

Frames where the remaining pair splits admit no majority and are excluded, as in
the SAGES analysis. What cannot be recovered is *which* rater dissented, so
per-rater drift is not computable here -- but the SAGES rater columns turned out
to be positional slots rather than identified people, so that analysis was
dropped there too.

**A separate finding this file records.** The released ``*_dataset_value`` column
is the mean of the three annotators, taking values 0, 1/3, 2/3 and 1, not the
majority vote. The Endoscapes technical report describes the ground truth as a
majority vote and the Scientific Data descriptor as an average of three
annotators; the data settles it. Comparing ``dataset_value`` against a binary
majority agrees on exactly the unanimous frames -- 71.65%, 83.99% and 70.73% for
C1, C2 and C3 -- which are the unanimous fractions themselves. Every published
result thresholds the mean; the distinction belongs in the comparability
section.

Usage:
    python eval/endoscapes_agreement.py \\
        --manifest metadata/endoscapes_frames.csv \\
        --split test --output-dir ../outputs/endoscapes_agreement \\
        --arm dinov2_b=../outputs/cvs-domain-ssl/probe/dinov2_b_endoscapes_mean:../cache/dinov2_b/endoscapes/test
"""

from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

CRITERIA = ("c1", "c2", "c3")


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-x))


def read_index(cache_dir: Path) -> list[str]:
    with open(cache_dir / "index.csv", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if [int(r["row"]) for r in rows] != list(range(len(rows))):
        raise ValueError(f"{cache_dir/'index.csv'} rows are not 0..N-1 in order.")
    return [r["sample_id"] for r in rows]


def load_scores(probe_dir: Path, n_rows: int, prefix: str) -> np.ndarray:
    files = sorted(probe_dir.glob(f"{prefix}_seed*.npz"))
    if not files:
        raise FileNotFoundError(f"No {prefix}_seed*.npz in {probe_dir}")
    stack = []
    for path in files:
        logits = np.load(path)["logits"]
        if logits.shape[0] != n_rows:
            raise ValueError(f"{path.name}: {logits.shape[0]} rows, cache has {n_rows}.")
        stack.append(sigmoid(logits))
    return np.mean(stack, axis=0)


def ceiling_from_counts(votes: np.ndarray) -> dict[str, Any]:
    """Leave-one-rater-out agreement, derived from the vote count.

    Every (held-out rater, majority of the other two) pair is enumerated. Pairs
    that split are dropped, since no majority exists. The result is the expected
    agreement of a randomly chosen annotator with the consensus of the other
    two, which is what the SAGES figure measures -- but derived rather than
    computed per rater, and labelled as such.
    """
    tp = fp = tn = fn = dropped = 0
    for v, n in zip(*np.unique(votes, return_counts=True)):
        v, n = int(v), int(n)
        for held in range(3):
            held_label = 1 if held < v else 0
            others = v - held_label
            if others == 1:                      # the pair splits; no majority
                dropped += n
                continue
            reference = 1 if others == 2 else 0
            if held_label == 1 and reference == 1: tp += n
            elif held_label == 1 and reference == 0: fp += n
            elif held_label == 0 and reference == 0: tn += n
            else: fn += n

    recall = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    prec = tp / max(tp + fp, 1)
    return {
        "bacc": 0.5 * (recall + spec),
        "f1": 2 * prec * recall / max(prec + recall, 1e-12),
        "recall": recall,
        "specificity": spec,
        "n_pairs": tp + fp + tn + fn,
        "n_dropped": dropped,
    }


def safe_auc(y: np.ndarray, s: np.ndarray) -> float:
    if y.size < 10 or np.unique(y).size < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


class Bootstrap:
    """Video-clustered resampling, as in eval/bootstrap_strata.py."""

    def __init__(self, videos: np.ndarray, n_boot: int, seed: int = 0) -> None:
        unique = np.unique(videos)
        by_video = {v: np.flatnonzero(videos == v) for v in unique}
        rng = np.random.default_rng(seed)
        self.replicates = [
            np.concatenate([by_video[v] for v in rng.choice(unique, unique.size, replace=True)])
            for _ in range(n_boot)
        ]
        self.n_videos = int(unique.size)

    def interval(self, statistic: Callable[[np.ndarray], float]) -> dict[str, float]:
        draws = np.array([statistic(r) for r in self.replicates], dtype=float)
        draws = draws[np.isfinite(draws)]
        if draws.size < 20:
            return {"ci_low": float("nan"), "ci_high": float("nan"), "n_draws": int(draws.size)}
        return {"ci_low": float(np.percentile(draws, 2.5)),
                "ci_high": float(np.percentile(draws, 97.5)),
                "n_draws": int(draws.size)}


def parse_arm(spec: str) -> tuple[str, Path, Path]:
    name, rest = spec.split("=", 1)
    probe, cache = rest.rsplit(":", 1)
    return name, Path(probe), Path(cache)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True)
    p.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    p.add_argument("--arm", action="append", default=[], type=parse_arm,
                   help="name=probe_dir:cache_dir (repeatable)")
    p.add_argument("--logits-prefix", default="test_logits")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--per-criterion", action="store_true",
                   help="report the stratified AUC and the pairwise counts for "
                        "C1, C2 and C3 separately. The criteria differ enough in "
                        "prevalence and in how the strata are balanced that the "
                        "mean over them can hide the pattern.")
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    meta = pd.read_csv(args.manifest)
    meta = meta[meta.is_cvs_annotated]
    if args.split != "all":
        meta = meta[meta.split == args.split]
    if meta.empty:
        raise SystemExit(f"No annotated rows for split={args.split}.")

    votes = {c: meta[f"{c}_manual_votes"].to_numpy(dtype=int) for c in CRITERIA}
    labels = {c: (votes[c] >= 2).astype(int) for c in CRITERIA}
    unanimous = {c: (votes[c] == 0) | (votes[c] == 3) for c in CRITERIA}

    results: dict[str, Any] = {"split": args.split, "n_frames": int(len(meta)),
                               "n_videos": int(meta.video_id.nunique())}

    print(f"Endoscapes, split={args.split}, {len(meta)} annotated frames, "
          f"{meta.video_id.nunique()} videos\n")

    print("Annotation ceiling, derived from vote counts")
    print(f"{'crit':<5}{'BAcc':>9}{'F1':>9}{'Recall':>9}{'Spec':>9}{'dropped':>10}")
    results["ceiling"] = {}
    for c in CRITERIA:
        r = ceiling_from_counts(votes[c])
        results["ceiling"][c] = r
        print(f"{c:<5}{r['bacc']:>9.4f}{r['f1']:>9.4f}{r['recall']:>9.4f}"
              f"{r['specificity']:>9.4f}{r['n_dropped']:>10d}")
    mean_bacc = float(np.mean([results["ceiling"][c]["bacc"] for c in CRITERIA]))
    print(f"{'mean':<5}{mean_bacc:>9.4f}")
    print("  Derived from counts, so this is the expected agreement of a randomly")
    print("  chosen annotator with the consensus of the other two. Rater identity")
    print("  is not released, so it cannot be computed per rater as on SAGES.")

    print(f"\nAgreement strata")
    print(f"{'crit':<5}{'unan':>7}{'cont':>7}{'prev_u':>9}{'prev_c':>9}{'%pos contested':>16}")
    results["strata"] = {}
    for c in CRITERIA:
        un, y = unanimous[c], labels[c]
        pos, pos_c = int(y.sum()), int((y & ~un).sum())
        row = {"n_unanimous": int(un.sum()), "n_contested": int((~un).sum()),
               "prevalence_unanimous": float(y[un].mean()),
               "prevalence_contested": float(y[~un].mean()),
               "n_positives": pos, "share_positives_contested": pos_c / max(pos, 1)}
        results["strata"][c] = row
        print(f"{c:<5}{row['n_unanimous']:>7d}{row['n_contested']:>7d}"
              f"{row['prevalence_unanimous']:>9.4f}{row['prevalence_contested']:>9.4f}"
              f"{100*row['share_positives_contested']:>15.1f}%")

    if not args.arm:
        out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
        with open(out / f"endoscapes_agreement_{args.split}.json", "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nwritten to {out}")
        return 0

    indexed = meta.set_index("sample_id")
    arms, ids_ref = {}, None
    for name, probe, cache in args.arm:
        ids = read_index(cache)
        if ids_ref is None:
            ids_ref = ids
        elif ids != ids_ref:
            raise SystemExit(f"{name} was scored on a different sample order.")
        arms[name] = load_scores(probe, len(ids), args.logits_prefix)

    assert ids_ref is not None
    missing = [s for s in ids_ref if s not in indexed.index]
    if missing:
        raise SystemExit(f"{len(missing)} sample_ids absent from the manifest, "
                         f"e.g. {missing[:3]}")
    aligned = indexed.loc[ids_ref].reset_index()
    votes = {c: aligned[f"{c}_manual_votes"].to_numpy(dtype=int) for c in CRITERIA}
    labels = {c: (votes[c] >= 2).astype(int) for c in CRITERIA}
    unanimous = {c: (votes[c] == 0) | (votes[c] == 3) for c in CRITERIA}
    all_rows = np.arange(len(aligned))
    boot = Bootstrap(aligned["video_id"].to_numpy(), args.n_boot)

    def stratum_auc(y, s, mask, rows):
        sel = rows[mask[rows]]
        return safe_auc(y[sel], s[sel]) if sel.size else float("nan")

    print(f"\nAUC by agreement stratum, mean over criteria "
          f"({boot.n_videos} videos, {args.n_boot} replicates)")
    print(f"{'arm':<22}{'all':>9}{'unanimous':>11}{'contested':>11}{'delta':>9}")
    results["arms"] = {}
    for name, probs in arms.items():
        vals = {}
        for stratum, mask_for in (("unanimous", lambda c: unanimous[c]),
                                  ("contested", lambda c: ~unanimous[c]),
                                  ("all", lambda c: np.ones(len(aligned), bool))):
            vals[stratum] = float(np.mean([
                stratum_auc(labels[c], probs[:, j], mask_for(c), all_rows)
                for j, c in enumerate(CRITERIA)]))
        results["arms"][name] = vals
        print(f"{name:<22}{vals['all']:>9.4f}{vals['unanimous']:>11.4f}"
              f"{vals['contested']:>11.4f}{vals['unanimous']-vals['contested']:>9.4f}")

    if len(arms) >= 2:
        counts = {"unanimous": 0, "contested": 0}
        results["pairwise"] = {}
        for a, b in combinations(arms, 2):
            pa, pb = arms[a], arms[b]
            for stratum, mask_for in (("unanimous", lambda c: unanimous[c]),
                                      ("contested", lambda c: ~unanimous[c])):
                def diff(rows, pa=pa, pb=pb, mask_for=mask_for):
                    vals = [stratum_auc(labels[c], pa[:, j], mask_for(c), rows)
                            - stratum_auc(labels[c], pb[:, j], mask_for(c), rows)
                            for j, c in enumerate(CRITERIA)]
                    vals = [v for v in vals if np.isfinite(v)]
                    return float(np.mean(vals)) if vals else float("nan")
                ci = boot.interval(diff)
                sig = np.isfinite(ci["ci_low"]) and ci["ci_low"] * ci["ci_high"] > 0
                counts[stratum] += int(sig)
                results["pairwise"].setdefault(f"{a}-{b}", {})[stratum] = {
                    "diff": diff(all_rows), **ci}
        total = len(list(combinations(arms, 2)))
        print(f"\nPairwise intervals excluding zero, of {total}:")
        print(f"  unanimous   {counts['unanimous']}")
        print(f"  contested   {counts['contested']}")
        print("  The SAGES equivalent was 16 of 36 and 0 of 36. A second dataset with a")
        print("  different annotation protocol and institution tests whether that holds.")

    if args.per_criterion:
        print(f"\nBy criterion")
        results["per_criterion"] = {}
        for j, c in enumerate(CRITERIA):
            un = unanimous[c]
            pos_u = int(labels[c][un].sum()); pos_c = int(labels[c][~un].sum())
            print(f"\n  {c.upper()}   unanimous {int(un.sum())} frames / {pos_u} positive"
                  f"   contested {int((~un).sum())} / {pos_c} positive")
            print(f"  {'arm':<20}{'unanimous':>11}{'contested':>11}{'delta':>9}")
            entry: dict[str, Any] = {"arms": {}, "n_pos_unanimous": pos_u,
                                     "n_pos_contested": pos_c}
            for name, probs in arms.items():
                u = stratum_auc(labels[c], probs[:, j], un, all_rows)
                k = stratum_auc(labels[c], probs[:, j], ~un, all_rows)
                entry["arms"][name] = {"unanimous": u, "contested": k, "delta": u - k}
                print(f"  {name:<20}{u:>11.4f}{k:>11.4f}{u-k:>9.4f}")

            counts = {"unanimous": 0, "contested": 0}
            entry["pairwise"] = {}
            for a, b in combinations(arms, 2):
                pa, pb = arms[a], arms[b]
                for stratum, mask in (("unanimous", un), ("contested", ~un)):
                    def diff(rows, pa=pa, pb=pb, mask=mask):
                        x = stratum_auc(labels[c], pa[:, j], mask, rows)
                        y = stratum_auc(labels[c], pb[:, j], mask, rows)
                        return x - y if np.isfinite(x) and np.isfinite(y) else float("nan")
                    ci = boot.interval(diff)
                    sig = np.isfinite(ci["ci_low"]) and ci["ci_low"] * ci["ci_high"] > 0
                    counts[stratum] += int(sig)
                    entry["pairwise"].setdefault(f"{a}-{b}", {})[stratum] = {
                        "diff": diff(all_rows), **ci}
            entry["n_significant"] = counts
            print(f"  pairs excluding zero, of {len(list(combinations(arms, 2)))}:"
                  f"  unanimous {counts['unanimous']}, contested {counts['contested']}")
            results["per_criterion"][c] = entry

        print(f"\n  A criterion whose unanimous stratum holds few positives cannot")
        print(f"  support the comparison there, whatever the encoders do. Read the")
        print(f"  positive counts in the header of each block before the counts below it.")

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    with open(out / f"endoscapes_agreement_{args.split}.json", "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwritten to {out/f'endoscapes_agreement_{args.split}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
