#!/usr/bin/env python3
"""Does the model recover annotator uncertainty it was never trained on?

Toolkit item 5. The obvious phrasing -- "when the model disagrees with the
majority, does it agree with the dissenting rater?" -- is vacuous: on a
contested frame the dissenter's label is by definition the complement of the
majority, so the two statements are the same statement. Three non-trivial
questions replace it.

**(a) Graded structure.** Training collapses three votes into a binary majority,
so a 1-vote frame and a 0-vote frame are given the identical target 0, and a
2-vote and a 3-vote frame the identical target 1. If the model nonetheless
ranks 1-vote frames above 0-vote frames, it has recovered information the
label discarded. Measured as AUC *within* a training class:

    AUC_neg : among frames with 0 or 1 votes, does the score rank 1 above 0?
    AUC_pos : among frames with 2 or 3 votes, does the score rank 3 above 2?

0.50 means the model treats the two identically, exactly as trained. Above 0.50
means the annotation uncertainty is visible in the representation.

**(b) Rater drift.** The model is scored against each rater individually. A
systematic preference for one rater means it has absorbed that annotator's
threshold rather than a shared standard.

**(c) The model as a fourth rater.** Each human was scored by leave-one-rater-
out: their label against the majority of the other two, ties dropped. The model
is put through the identical protocol against all three pairs, which places it
on the same axis as the surgeons rather than on an unbounded 0-1 scale.

Intervals come from a cluster bootstrap over videos. The 18 frames of one
procedure share patient, anatomy, camera and illumination, so resampling frames
would understate every interval by roughly sqrt(18).

Usage:
    python eval/dissent_alignment.py \\
        --manifest metadata/sages_frames_internal_split.csv \\
        --arm dinov3_b=../outputs/cvs-domain-ssl/probe/dinov3_b_sages_mean:../cache/dinov3_b/sages/val \\
        --output-dir ../outputs/dissent_alignment
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

CRITERIA = ("c1", "c2", "c3")
RATERS = (1, 2, 3)


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-x))


def read_index(cache_dir: Path) -> list[str]:
    with open(cache_dir / "index.csv", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if [int(r["row"]) for r in rows] != list(range(len(rows))):
        raise ValueError(f"{cache_dir/'index.csv'} rows are not 0..N-1 in order.")
    return [r["sample_id"] for r in rows]


def load_scores(probe_dir: Path, n_rows: int) -> np.ndarray:
    """Mean sigmoid score over seeds, [N, 3]."""
    files = sorted(probe_dir.glob("val_logits_seed*.npz"))
    if not files:
        raise FileNotFoundError(f"No val_logits_seed*.npz in {probe_dir}")
    stack = []
    for path in files:
        logits = np.load(path)["logits"]
        if logits.shape[0] != n_rows:
            raise ValueError(f"{path.name}: {logits.shape[0]} rows, cache has {n_rows}.")
        stack.append(sigmoid(logits))
    return np.mean(stack, axis=0)


def prevalence_matched_threshold(scores: np.ndarray, prevalence: float) -> float:
    if not 0.0 < prevalence < 1.0:
        return float("inf")
    return float(np.quantile(scores, 1.0 - prevalence))


def cluster_bootstrap(
    statistic, videos: np.ndarray, n_boot: int, seed: int = 0
) -> tuple[float, float]:
    """Percentile interval from resampling whole videos with replacement."""
    unique = np.unique(videos)
    index_by_video = {v: np.flatnonzero(videos == v) for v in unique}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        picked = rng.choice(unique, size=unique.size, replace=True)
        rows = np.concatenate([index_by_video[v] for v in picked])
        value = statistic(rows)
        if value is not None and not np.isnan(value):
            draws.append(value)
    if len(draws) < 20:
        return float("nan"), float("nan")
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def safe_auc(y: np.ndarray, s: np.ndarray) -> float | None:
    if y.size < 10 or np.unique(y).size < 2:
        return None
    return float(roc_auc_score(y, s))


def graded_structure(
    votes: np.ndarray, scores: np.ndarray, videos: np.ndarray, n_boot: int
) -> dict[str, Any]:
    """Within each training class, does the score track the vote count?"""
    out: dict[str, Any] = {}
    for name, (low, high) in (("neg", (0, 1)), ("pos", (2, 3))):
        mask = (votes == low) | (votes == high)
        y = (votes[mask] == high).astype(int)
        s = scores[mask]
        auc = safe_auc(y, s)
        if auc is None:
            out[name] = {"auc": float("nan"), "n_low": 0, "n_high": 0}
            continue

        sub_videos = videos[mask]

        def stat(rows, y=y, s=s):
            return safe_auc(y[rows], s[rows])

        lo, hi = cluster_bootstrap(stat, sub_videos, n_boot)
        out[name] = {
            "auc": auc,
            "ci_low": lo,
            "ci_high": hi,
            "n_low": int((votes == low).sum()),
            "n_high": int((votes == high).sum()),
            "mean_score_low": float(scores[votes == low].mean()),
            "mean_score_high": float(scores[votes == high].mean()),
        }
    return out


def rater_drift(df: pd.DataFrame, criterion: str, pred: np.ndarray) -> dict[str, float]:
    """Model balanced accuracy against each rater taken individually."""
    out = {}
    for r in RATERS:
        y = df[f"{criterion}_rater{r}"].to_numpy(dtype=int)
        out[f"rater{r}"] = (
            float(balanced_accuracy_score(y, pred)) if np.unique(y).size > 1 else float("nan")
        )
    values = [v for v in out.values() if not np.isnan(v)]
    out["spread"] = float(max(values) - min(values)) if values else float("nan")
    return out


def model_as_fourth_rater(df: pd.DataFrame, criterion: str, pred: np.ndarray) -> dict[str, Any]:
    """Score the model by the protocol used for the humans: against pair majorities."""
    votes = df[[f"{criterion}_rater{r}" for r in RATERS]].to_numpy(dtype=int)
    per_pair = []
    for a, b in ((0, 1), (0, 2), (1, 2)):
        agreed = votes[:, a] == votes[:, b]
        reference = votes[agreed, a]
        if np.unique(reference).size < 2:
            continue
        per_pair.append(
            {
                "pair": f"r{a+1}+r{b+1}",
                "n": int(agreed.sum()),
                "bacc": float(balanced_accuracy_score(reference, pred[agreed])),
            }
        )
    if not per_pair:
        return {"bacc": float("nan"), "per_pair": []}
    return {"bacc": float(np.mean([p["bacc"] for p in per_pair])), "per_pair": per_pair}


def human_ceiling_bacc(df: pd.DataFrame, criterion: str) -> float:
    """The same leave-one-rater-out figure, recomputed here for side-by-side display."""
    votes = df[[f"{criterion}_rater{r}" for r in RATERS]].to_numpy(dtype=int)
    scores = []
    for held in range(3):
        others = [j for j in range(3) if j != held]
        pair = votes[:, others]
        agreed = pair[:, 0] == pair[:, 1]
        reference = pair[agreed, 0]
        if np.unique(reference).size < 2:
            continue
        scores.append(float(balanced_accuracy_score(reference, votes[agreed, held])))
    return float(np.mean(scores)) if scores else float("nan")


def analyse_arm(
    name: str, probe_dir: Path, cache_dir: Path, meta: pd.DataFrame, n_boot: int
) -> dict[str, Any]:
    sample_ids = read_index(cache_dir)
    indexed = meta.set_index("sample_id")
    unknown = [s for s in sample_ids if s not in indexed.index]
    if unknown:
        raise ValueError(f"{len(unknown)} sample_ids absent from manifest, e.g. {unknown[:3]}")
    df = indexed.loc[sample_ids].reset_index()

    probs = load_scores(probe_dir, len(sample_ids))
    expected = df[[f"{c}_consensus" for c in CRITERIA]].to_numpy(dtype=float)
    videos = df["video_id"].to_numpy()

    result: dict[str, Any] = {"arm": name, "n_frames": len(df), "criteria": {}}
    for i, criterion in enumerate(CRITERIA):
        votes = df[[f"{criterion}_rater{r}" for r in RATERS]].to_numpy(dtype=int).sum(axis=1)
        scores = probs[:, i]
        y = expected[:, i].astype(int)
        threshold = prevalence_matched_threshold(scores, float(y.mean()))
        pred = (scores >= threshold).astype(int)

        result["criteria"][criterion] = {
            "graded": graded_structure(votes, scores, videos, n_boot),
            "drift": rater_drift(df, criterion, pred),
            "fourth_rater": model_as_fourth_rater(df, criterion, pred),
            "human_ceiling_bacc": human_ceiling_bacc(df, criterion),
        }
    return result


def report(arms: list[dict[str, Any]]) -> None:
    print("\n(a) Graded structure -- can the score separate frames the label merged?")
    print(f"{'arm':<20}{'crit':<5}{'0 vs 1 vote':>26}{'2 vs 3 votes':>26}")
    for arm in arms:
        for criterion in CRITERIA:
            g = arm["criteria"][criterion]["graded"]
            cells = []
            for key in ("neg", "pos"):
                v = g.get(key, {})
                if np.isnan(v.get("auc", float("nan"))):
                    cells.append(f"{'n/a':>26}")
                else:
                    cells.append(
                        f"{v['auc']:>10.4f} [{v.get('ci_low', float('nan')):.3f},"
                        f"{v.get('ci_high', float('nan')):.3f}]"
                    )
            print(f"{arm['arm']:<20}{criterion:<5}{cells[0]}{cells[1]}")
    print("  0.500 = the model treats the two exactly as the training label did.")
    print("  Above 0.500 with a CI clear of 0.5 = uncertainty survived the collapse to binary.")

    print("\n(b) Rater drift -- model BAcc against each rater individually")
    print(f"{'arm':<20}{'crit':<5}{'rater1':>9}{'rater2':>9}{'rater3':>9}{'spread':>9}")
    for arm in arms:
        for criterion in CRITERIA:
            d = arm["criteria"][criterion]["drift"]
            print(f"{arm['arm']:<20}{criterion:<5}{d['rater1']:>9.4f}"
                  f"{d['rater2']:>9.4f}{d['rater3']:>9.4f}{d['spread']:>9.4f}")

    print("\n(c) Model as a fourth rater -- identical protocol to the human ceiling")
    print(f"{'arm':<20}{'crit':<5}{'model':>9}{'human':>9}{'gap':>9}")
    for arm in arms:
        for criterion in CRITERIA:
            c = arm["criteria"][criterion]
            m, h = c["fourth_rater"]["bacc"], c["human_ceiling_bacc"]
            print(f"{arm['arm']:<20}{criterion:<5}{m:>9.4f}{h:>9.4f}{m-h:>9.4f}")


def parse_arm(spec: str) -> tuple[str, Path, Path]:
    if "=" not in spec or ":" not in spec.split("=", 1)[1]:
        raise argparse.ArgumentTypeError(f"--arm must be name=probe_dir:cache_dir, got {spec!r}")
    name, rest = spec.split("=", 1)
    probe_dir, cache_dir = rest.rsplit(":", 1)
    return name, Path(probe_dir), Path(cache_dir)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True)
    p.add_argument("--arm", action="append", default=[], type=parse_arm, required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--bootstrap", type=int, default=1000)
    args = p.parse_args()

    meta = pd.read_csv(args.manifest)
    arms = []
    for name, probe_dir, cache_dir in args.arm:
        try:
            arms.append(analyse_arm(name, probe_dir, cache_dir, meta, args.bootstrap))
        except (FileNotFoundError, ValueError) as exc:
            print(f"[skipped] {name}: {exc}")
    if not arms:
        print("No arms analysed.")
        return 1

    report(arms)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "dissent_alignment.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"bootstrap": args.bootstrap, "arms": arms}, fh, indent=2)
    print(f"\nwritten to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
