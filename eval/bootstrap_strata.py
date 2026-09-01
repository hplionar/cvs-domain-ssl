#!/usr/bin/env python3
"""Video-clustered bootstrap intervals for the agreement-stratified comparison.

The stratified evaluation reports probe-seed standard deviations, which are
small -- median around 0.006 -- because the encoder is frozen and the features
cached, so across seeds the inputs are byte-identical and only head
initialisation and batch order vary. That is the narrowest source of variance in
the experiment. Two larger ones are unmeasured:

* **Frame sampling.** The contested stratum is 205-289 frames from 70 videos.
  The sampling standard error of an AUC at that size is of order 0.04, six times
  the seed SD, and nothing in the current output reflects that the validation
  split is one draw of 70 procedures.
* **Pretraining.** For adapted arms, n = 1.

This module addresses the first. It resamples whole **videos** with replacement,
not frames: the 18 frames of one procedure share patient, anatomy, camera and
illumination, so resampling frames would treat 18 correlated observations as
independent and understate every interval by roughly sqrt(18).

**Paired differences, not marginals.** Arms are scored on identical frames, so
a bootstrap replicate that happens to draw easy videos raises every arm at once.
Taking the interval on the difference between two arms cancels that shared
component and gives intervals far tighter -- and far more honest -- than
comparing two marginal intervals for overlap. Two marginal 95% intervals can
overlap while the paired difference is unambiguously non-zero.

Three quantities get intervals:

    delta       one arm's unanimous AUC minus its contested AUC
    spread      the between-arm range within a stratum, and the ratio of the
                two ranges -- the headline claim
    pairwise    arm A minus arm B within one stratum

Usage:
    python eval/bootstrap_strata.py \\
        --manifest metadata/sages_frames_internal_split.csv \\
        --split val --n-boot 2000 --output-dir ../outputs/rater_agreement \\
        --arm dinov3_b=../outputs/cvs-domain-ssl/probe/dinov3_b_sages_mean:../cache/dinov3_b/sages/val \\
        --arm mae_b=../outputs/cvs-domain-ssl/probe/mae_b_sages_mean:../cache/mae_b/sages/val
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
from sklearn.metrics import roc_auc_score

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
    """Mean probability over seeds, [N, 3].

    Averaging over seeds first is deliberate: the bootstrap is about sampling
    variability of the videos, and carrying seed variance into it would conflate
    two sources that we want reported separately.
    """
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


def safe_auc(y: np.ndarray, s: np.ndarray) -> float:
    if y.size < 10 or np.unique(y).size < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


class Bootstrap:
    """Resamples videos once, then evaluates any statistic on the same draws.

    Using one set of replicate index arrays for every statistic keeps the
    comparisons coherent: a difference between two arms is computed on the same
    resampled videos that produced each arm's marginal value.
    """

    def __init__(self, videos: np.ndarray, n_boot: int, seed: int = 0) -> None:
        unique = np.unique(videos)
        by_video = {v: np.flatnonzero(videos == v) for v in unique}
        rng = np.random.default_rng(seed)
        self.replicates: list[np.ndarray] = []
        for _ in range(n_boot):
            picked = rng.choice(unique, size=unique.size, replace=True)
            self.replicates.append(np.concatenate([by_video[v] for v in picked]))
        self.n_videos = int(unique.size)

    def interval(self, statistic: Callable[[np.ndarray], float]) -> dict[str, float]:
        draws = np.array([statistic(rows) for rows in self.replicates], dtype=float)
        draws = draws[np.isfinite(draws)]
        if draws.size < 20:
            return {"ci_low": float("nan"), "ci_high": float("nan"),
                    "boot_sd": float("nan"), "n_draws": int(draws.size)}
        return {
            "ci_low": float(np.percentile(draws, 2.5)),
            "ci_high": float(np.percentile(draws, 97.5)),
            "boot_sd": float(draws.std(ddof=1)),
            "n_draws": int(draws.size),
        }


def stratum_auc(y: np.ndarray, s: np.ndarray, mask: np.ndarray, rows: np.ndarray,
                cap: int | None = None, rng: np.random.Generator | None = None) -> float:
    """AUC over the rows of one stratum, optionally capped in size.

    ``cap`` subsamples the selected rows without replacement, which is used only
    to compare two strata at equal frame counts. The subsample is drawn after
    the videos have been resampled, so the clustering is preserved.
    """
    sel = rows[mask[rows]]
    if cap is not None and sel.size > cap:
        sel = (rng or np.random.default_rng(0)).choice(sel, size=cap, replace=False)
    return safe_auc(y[sel], s[sel]) if sel.size else float("nan")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True)
    p.add_argument("--arm", action="append", required=True,
                   help="name=probe_dir:cache_dir (repeatable)")
    p.add_argument("--split", default="val")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--match-power", action="store_true",
                   help="additionally report the unanimous stratum subsampled to the "
                        "size of the contested stratum, so that the two are compared "
                        "at equal frame counts")
    p.add_argument("--per-criterion", action="store_true",
                   help="report the delta separately for C1, C2 and C3 rather than "
                        "only their mean")
    args = p.parse_args()

    meta = pd.read_csv(args.manifest).set_index("sample_id")

    arms: dict[str, dict[str, Any]] = {}
    reference_ids: list[str] | None = None
    for spec in args.arm:
        name, rest = spec.split("=", 1)
        probe_dir, cache_dir = rest.rsplit(":", 1)
        ids = read_index(Path(cache_dir))
        if reference_ids is None:
            reference_ids = ids
        elif ids != reference_ids:
            raise SystemExit(
                f"{name} was scored on a different sample order from the first arm. "
                f"Paired differences require identical frames in identical order."
            )
        arms[name] = {"probs": load_scores(Path(probe_dir), len(ids))}

    assert reference_ids is not None
    df = meta.loc[reference_ids].reset_index()
    videos = df["video_id"].to_numpy()
    boot = Bootstrap(videos, args.n_boot, args.seed)
    print(f"{len(df)} frames, {boot.n_videos} videos, {args.n_boot} replicates, "
          f"{len(arms)} arms\n")

    labels, unanimous = {}, {}
    for j, criterion in enumerate(CRITERIA):
        votes = df[[f"{criterion}_rater{r}" for r in RATERS]].to_numpy(int).sum(axis=1)
        labels[criterion] = (votes >= 2).astype(int)
        unanimous[criterion] = (votes == 0) | (votes == 3)

    all_rows = np.arange(len(df))
    results: dict[str, Any] = {"n_boot": args.n_boot, "n_videos": boot.n_videos,
                               "n_frames": len(df), "arms": {}, "pairwise": {}, "spread": {}}

    # --- per arm: the unanimous-minus-contested delta ----------------------
    print("Delta = unanimous AUC - contested AUC, mean over criteria")
    print(f"{'arm':<20}{'unan':>9}{'cont':>9}{'delta':>9}{'95% CI':>22}")
    for name, arm in arms.items():
        probs = arm["probs"]

        def delta(rows: np.ndarray, probs=probs) -> float:
            vals = []
            for j, c in enumerate(CRITERIA):
                u = stratum_auc(labels[c], probs[:, j], unanimous[c], rows)
                k = stratum_auc(labels[c], probs[:, j], ~unanimous[c], rows)
                if np.isfinite(u) and np.isfinite(k):
                    vals.append(u - k)
            return float(np.mean(vals)) if vals else float("nan")

        point_u = float(np.mean([stratum_auc(labels[c], probs[:, j], unanimous[c], all_rows)
                                 for j, c in enumerate(CRITERIA)]))
        point_k = float(np.mean([stratum_auc(labels[c], probs[:, j], ~unanimous[c], all_rows)
                                 for j, c in enumerate(CRITERIA)]))
        ci = boot.interval(delta)
        arm["delta"] = {"unanimous": point_u, "contested": point_k,
                        "delta": point_u - point_k, **ci}
        results["arms"][name] = arm["delta"]
        print(f"{name:<20}{point_u:>9.4f}{point_k:>9.4f}{point_u - point_k:>9.4f}"
              f"   [{ci['ci_low']:+.4f}, {ci['ci_high']:+.4f}]")

    # --- between-arm spread, and the ratio of the two spreads --------------
    if len(arms) >= 2:
        def spread(stratum: str) -> Callable[[np.ndarray], float]:
            mask_for = (lambda c: unanimous[c]) if stratum == "unanimous" else (lambda c: ~unanimous[c])

            def stat(rows: np.ndarray) -> float:
                means = []
                for arm in arms.values():
                    vals = [stratum_auc(labels[c], arm["probs"][:, j], mask_for(c), rows)
                            for j, c in enumerate(CRITERIA)]
                    if all(np.isfinite(v) for v in vals):
                        means.append(float(np.mean(vals)))
                return float(max(means) - min(means)) if len(means) == len(arms) else float("nan")
            return stat

        def ratio(rows: np.ndarray) -> float:
            u, k = spread("unanimous")(rows), spread("contested")(rows)
            return u / k if np.isfinite(u) and np.isfinite(k) and k > 1e-9 else float("nan")

        print(f"\nBetween-arm spread (max - min of the per-arm mean AUC)")
        for stratum in ("unanimous", "contested"):
            point = spread(stratum)(all_rows)
            ci = boot.interval(spread(stratum))
            results["spread"][stratum] = {"point": point, **ci}
            print(f"  {stratum:<12}{point:>8.4f}   [{ci['ci_low']:.4f}, {ci['ci_high']:.4f}]")
        point_ratio = ratio(all_rows)
        ci = boot.interval(ratio)
        results["spread"]["ratio"] = {"point": point_ratio, **ci}
        print(f"  {'ratio':<12}{point_ratio:>8.2f}x  [{ci['ci_low']:.2f}, {ci['ci_high']:.2f}]")
        print("  A lower bound above 1.0 means the spread really is larger on")
        print("  unanimous frames, not an artefact of which videos were drawn.")

    # --- pairwise differences within each stratum --------------------------
    if len(arms) >= 2:
        print(f"\nPairwise AUC differences, paired on identical frames")
        print(f"{'pair':<34}{'stratum':<12}{'diff':>9}{'95% CI':>22}  sig")
        for a, b in combinations(arms, 2):
            pa, pb = arms[a]["probs"], arms[b]["probs"]
            for stratum in ("unanimous", "contested"):
                mask_for = (lambda c: unanimous[c]) if stratum == "unanimous" else (lambda c: ~unanimous[c])

                def diff(rows: np.ndarray, pa=pa, pb=pb, mask_for=mask_for) -> float:
                    vals = []
                    for j, c in enumerate(CRITERIA):
                        x = stratum_auc(labels[c], pa[:, j], mask_for(c), rows)
                        y = stratum_auc(labels[c], pb[:, j], mask_for(c), rows)
                        if np.isfinite(x) and np.isfinite(y):
                            vals.append(x - y)
                    return float(np.mean(vals)) if vals else float("nan")

                point = diff(all_rows)
                ci = boot.interval(diff)
                sig = "*" if np.isfinite(ci["ci_low"]) and ci["ci_low"] * ci["ci_high"] > 0 else ""
                results["pairwise"].setdefault(f"{a}-{b}", {})[stratum] = {"diff": point, **ci}
                print(f"{a+' - '+b:<34}{stratum:<12}{point:>+9.4f}"
                      f"   [{ci['ci_low']:+.4f}, {ci['ci_high']:+.4f}]  {sig}")
        print("  * = the interval excludes zero. Paired on identical frames, so this")
        print("    is a much sharper test than checking whether marginal intervals overlap.")

    # --- per criterion -----------------------------------------------------
    if args.per_criterion:
        print(f"\nDelta by criterion")
        print(f"{'arm':<20}{'crit':<5}{'unan':>9}{'cont':>9}{'delta':>9}{'95% CI':>22}")
        results["per_criterion"] = {}
        for name, arm in arms.items():
            probs = arm["probs"]
            results["per_criterion"][name] = {}
            for j, c in enumerate(CRITERIA):
                def delta_c(rows: np.ndarray, probs=probs, j=j, c=c) -> float:
                    u = stratum_auc(labels[c], probs[:, j], unanimous[c], rows)
                    k = stratum_auc(labels[c], probs[:, j], ~unanimous[c], rows)
                    return u - k if np.isfinite(u) and np.isfinite(k) else float("nan")

                pu = stratum_auc(labels[c], probs[:, j], unanimous[c], all_rows)
                pk = stratum_auc(labels[c], probs[:, j], ~unanimous[c], all_rows)
                ci = boot.interval(delta_c)
                results["per_criterion"][name][c] = {
                    "unanimous": pu, "contested": pk, "delta": pu - pk, **ci}
                print(f"{name:<20}{c:<5}{pu:>9.4f}{pk:>9.4f}{pu - pk:>9.4f}"
                      f"   [{ci['ci_low']:+.4f}, {ci['ci_high']:+.4f}]")

    # --- power-matched control ---------------------------------------------
    if args.match_power and len(arms) >= 2:
        # The contested stratum is roughly a quarter the size of the unanimous
        # one, so the absence of distinguishable pairs there could reflect
        # sample size rather than an absence of difference. Subsampling the
        # unanimous stratum to the contested size, inside each replicate,
        # separates the two explanations.
        caps = {c: int((~unanimous[c]).sum()) for c in CRITERIA}
        print(f"\nPower-matched control: unanimous subsampled to the contested size")
        print(f"  frames per criterion  " + "  ".join(f"{c} {caps[c]}" for c in CRITERIA))

        rng = np.random.default_rng(args.seed + 1)

        def spread_capped(rows: np.ndarray) -> float:
            means = []
            for arm in arms.values():
                vals = [stratum_auc(labels[c], arm["probs"][:, j], unanimous[c], rows,
                                    cap=caps[c], rng=rng)
                        for j, c in enumerate(CRITERIA)]
                if all(np.isfinite(v) for v in vals):
                    means.append(float(np.mean(vals)))
            return float(max(means) - min(means)) if len(means) == len(arms) else float("nan")

        point = spread_capped(all_rows)
        ci = boot.interval(spread_capped)
        results["spread"]["unanimous_power_matched"] = {"point": point, **ci}
        print(f"  {'spread':<12}{point:>8.4f}   [{ci['ci_low']:.4f}, {ci['ci_high']:.4f}]"
              f"   (full unanimous {results['spread']['unanimous']['point']:.4f})")

        significant = 0
        for a, b in combinations(arms, 2):
            pa, pb = arms[a]["probs"], arms[b]["probs"]

            def diff_capped(rows: np.ndarray, pa=pa, pb=pb) -> float:
                vals = []
                for j, c in enumerate(CRITERIA):
                    x = stratum_auc(labels[c], pa[:, j], unanimous[c], rows,
                                    cap=caps[c], rng=rng)
                    y = stratum_auc(labels[c], pb[:, j], unanimous[c], rows,
                                    cap=caps[c], rng=rng)
                    if np.isfinite(x) and np.isfinite(y):
                        vals.append(x - y)
                return float(np.mean(vals)) if vals else float("nan")

            ci = boot.interval(diff_capped)
            if np.isfinite(ci["ci_low"]) and ci["ci_low"] * ci["ci_high"] > 0:
                significant += 1
            results["pairwise"].setdefault(f"{a}-{b}", {})["unanimous_power_matched"] = {
                "diff": diff_capped(all_rows), **ci}

        total = len(list(combinations(arms, 2)))
        full_u = sum(1 for p in results["pairwise"].values()
                     if p.get("unanimous") and
                     p["unanimous"]["ci_low"] * p["unanimous"]["ci_high"] > 0)
        full_c = sum(1 for p in results["pairwise"].values()
                     if p.get("contested") and
                     p["contested"]["ci_low"] * p["contested"]["ci_high"] > 0)
        print(f"\n  pairwise intervals excluding zero, out of {total}:")
        print(f"    unanimous, full            {full_u}")
        print(f"    unanimous, size-matched    {significant}")
        print(f"    contested                  {full_c}")
        print("  If the size-matched count stays well above the contested count, the")
        print("  absence of distinguishable pairs on contested frames is not explained")
        print("  by the size of that stratum.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"bootstrap_{args.split}.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwritten to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
