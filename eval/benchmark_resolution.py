#!/usr/bin/env python3
"""The resolution of a benchmark: the smallest difference it returns reliably.

A benchmark is an instrument. Like any instrument it has a resolution, and
differences below it are not measurements. This script estimates that resolution
directly, from the same bootstrap the project already uses to compare arms.

**The estimate.** For every pair of arms, the paired difference in mAP is
resampled over videos. The half-width of the resulting interval is the smallest
difference that pair could have resolved. The median half-width across all pairs
is the instrument's resolution at that test size.

Paired rather than independent, because the two arms are scored on the same
videos and variation common to both cancels. Clustered by video rather than by
frame, because the frames of one procedure share patient, anatomy, camera and
illumination, and treating them as independent would understate the interval by
roughly the square root of the frames per video.

**Why this matters more than a significance test.** A test answers whether two
particular arms differ. The resolution answers what any future comparison on
this benchmark can establish, which is the quantity a reader needs to interpret
a published leaderboard. The top four published methods on Endoscapes span 0.060
mAP; whether that is four distinguishable systems or one depends entirely on a
number nobody has reported.

**Subsampling.** The resolution depends on the number of videos, and the
dependence is the point: it separates what more data would fix from what it
would not. Estimating at several test sizes shows how the curve falls and lets a
reader ask what a benchmark of a given size can support.

Usage:
    python eval/benchmark_resolution.py \\
        --probe-root ../outputs/cvs-domain-ssl/probe \\
        --cache-root ../cache \\
        --manifest metadata/sages_frames_official_test.csv \\
        --logits-prefix test_logits_official \\
        --cache-split sages_official/test \\
        --output-dir ../outputs/resolution
"""

from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def read_index(cache: Path) -> list[str]:
    with open(cache / "index.csv", newline="", encoding="utf-8") as fh:
        return [r["sample_id"] for r in csv.DictReader(fh)]


def load_arm(probe_dir: Path, prefix: str) -> np.ndarray | None:
    """Sigmoid scores per seed, [S, N, 3].

    Kept per seed rather than averaged: averaging before resampling removes
    seed variance from the interval, so the resolution it reports is the
    video-sampling component alone. Which seeds enter each replicate is decided
    by `seed_mode` in resolution_at().
    """
    files = sorted(probe_dir.glob(f"{prefix}_seed*.npz"))
    if not files:
        return None
    scores = []
    for path in files:
        payload = np.load(path)
        scores.append(1.0 / (1.0 + np.exp(-payload["logits"])))
    return np.stack(scores, axis=0)


def load_finetune_arm(root: Path, name: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Sigmoid scores per seed [S, N, 3] and targets [N, 3] for a fine-tuned arm.

    Seed runs live in sibling directories <name>_seed<k>; <name> itself is seed 0.
    Targets are read from the first file and checked against every other, which
    catches a run scored against a different manifest.
    """
    dirs = [root / name] + sorted(root.glob(f"{name}_seed*"))
    scores, targets = [], None
    for d in dirs:
        f = d / "logits_official_test.npz"
        if not f.is_file():
            continue
        payload = np.load(f)
        scores.append(1.0 / (1.0 + np.exp(-payload["logits"])))
        t = payload["targets"].astype(int)
        if targets is None:
            targets = t
        elif not np.array_equal(t, targets):
            raise SystemExit(f"{f}: targets differ from {dirs[0]}; not the same split")
    if not scores:
        return None
    return np.stack(scores, axis=0), targets


def mean_ap(y: np.ndarray, p: np.ndarray) -> float:
    """Macro-averaged AP, matching the challenge scorer and the project's own."""
    out = []
    for c in range(y.shape[1]):
        if np.unique(y[:, c]).size > 1:
            out.append(average_precision_score(y[:, c], p[:, c]))
    return float(np.mean(out)) if out else float("nan")


def resolution_at(
    scores: dict[str, np.ndarray],
    targets: np.ndarray,
    video_of: np.ndarray,
    n_videos: int | None,
    n_boot: int,
    rng: np.random.Generator,
    seed_mode: str = "inside",
) -> dict[str, Any]:
    """Median half-width of the paired difference interval, over all arm pairs.

    With `n_videos`, each replicate first draws that many videos without
    replacement and then resamples them with replacement, which estimates the
    resolution a benchmark of that size would have. Without it, all videos are
    resampled, which estimates the resolution of the benchmark as released.

    seed_mode decides which variance the interval carries:
      "mean"      scores averaged over seeds, videos resampled -- the video
                  sampling component alone (the previous behaviour);
      "inside"    one seed drawn per arm per replicate, videos resampled --
                  both components, the resolution as it applies to a
                  retrained comparison;
      "seed_only" videos held fixed, one seed drawn per arm -- the seed
                  component alone.
    With a single seed per arm, "inside" and "mean" coincide for that arm.
    """
    videos = np.unique(video_of)
    rows = {v: np.flatnonzero(video_of == v) for v in videos}
    names = sorted(scores)
    mean_scores = {n: scores[n].mean(axis=0) for n in names}
    all_idx = np.arange(video_of.size)

    widths: dict[tuple[str, str], list[float]] = {p: [] for p in combinations(names, 2)}
    for _ in range(n_boot):
        if seed_mode == "seed_only":
            idx = all_idx
        else:
            pool = (rng.choice(videos, n_videos, replace=False)
                    if n_videos and n_videos < videos.size else videos)
            draw = rng.choice(pool, pool.size, replace=True)
            idx = np.concatenate([rows[v] for v in draw])
        y = targets[idx]
        aps = {}
        for n in names:
            if seed_mode == "mean":
                aps[n] = mean_ap(y, mean_scores[n][idx])
            else:
                s = rng.integers(scores[n].shape[0])
                aps[n] = mean_ap(y, scores[n][s][idx])
        for a, b in widths:
            widths[(a, b)].append(aps[a] - aps[b])

    half = {}
    for pair, draws in widths.items():
        d = np.asarray(draws)
        lo, hi = np.percentile(d, 2.5), np.percentile(d, 97.5)
        half[f"{pair[0]}|{pair[1]}"] = {
            "half_width": float((hi - lo) / 2),
            "ci": [float(lo), float(hi)],
            "excludes_zero": bool(lo > 0 or hi < 0),
            # Minimum detectable effect at 80% power, two-sided 5%: the
            # effect whose interval would exclude zero four times in five.
            "mde_80": float(2.8 * np.std(d)),
            "point": float(np.mean(d)),
        }

    hw = np.array([v["half_width"] for v in half.values()])
    return {
        "n_videos": int(n_videos or videos.size),
        "seed_mode": seed_mode,
        "seeds_per_arm": {n: int(scores[n].shape[0]) for n in names},
        "n_pairs": len(half),
        "resolution_median": float(np.median(hw)),
        "resolution_q25": float(np.percentile(hw, 25)),
        "resolution_q75": float(np.percentile(hw, 75)),
        "pairs_resolved": int(sum(v["excludes_zero"] for v in half.values())),
        "pairs": half,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--probe-root", default=None)
    p.add_argument("--cache-root", default=None)
    p.add_argument("--finetune-root", default=None,
                   help="outputs/finetune; use with --finetune-arms instead of "
                        "--probe-root/--cache-root")
    p.add_argument("--finetune-arms", nargs="+", default=None,
                   help="run names; each is <name> plus every <name>_seed*")
    p.add_argument("--manifest", required=True)
    p.add_argument("--logits-prefix", default="test_logits_official")
    p.add_argument("--cache-split", default="sages_official/test")
    p.add_argument("--arms", nargs="+", default=None,
                   help="probe directory names. Default: every directory under "
                        "--probe-root holding logits with the given prefix.")
    p.add_argument("--subsample", type=int, nargs="+", default=[40, 70, 150, 300],
                   help="test sizes in videos. The dependence on size is the "
                        "point: it separates what more data would fix from what "
                        "it would not.")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    meta = pd.read_csv(args.manifest).set_index("sample_id")

    scores, sample_ids, targets = {}, None, None
    if args.finetune_root:
        if not args.finetune_arms:
            raise SystemExit("--finetune-root needs --finetune-arms")
        sample_ids = list(meta.index)  # score_finetune walks the manifest in file order
        for name in args.finetune_arms:
            loaded = load_finetune_arm(Path(args.finetune_root), name)
            if loaded is None:
                print(f"  no logits for {name}"); continue
            s, t = loaded
            if s.shape[1] != len(sample_ids):
                raise SystemExit(f"{name}: {s.shape[1]} rows against {len(sample_ids)} manifest rows")
            if targets is None:
                targets = t
            elif not np.array_equal(t, targets):
                raise SystemExit(f"{name}: targets differ from the first arm")
            scores[name] = s
        candidates = []
        probe_root = cache_root = None
    else:
        probe_root, cache_root = Path(args.probe_root), Path(args.cache_root)
        candidates = (args.arms if args.arms else
                      sorted(d.name for d in probe_root.iterdir() if d.is_dir()))
    for name in candidates:
        s = load_arm(probe_root / name, args.logits_prefix)
        if s is None:
            continue
        arm = name.replace("_sages_mean", "").replace("_mean", "")
        cache = cache_root / arm / args.cache_split
        if not (cache / "index.csv").is_file():
            print(f"  no cache for {arm}"); continue
        ids = read_index(cache)
        if sample_ids is None:
            sample_ids = ids
            rows = meta.loc[ids]
            targets = rows[["c1_consensus", "c2_consensus", "c3_consensus"]].to_numpy(int)
        elif ids != sample_ids:
            print(f"  {arm} holds a different sample order; skipped"); continue
        if s.shape[0] != len(ids):
            print(f"  {arm}: {s.shape[0]} rows against {len(ids)} cached; skipped")
            continue
        scores[arm] = s

    if len(scores) < 2:
        raise SystemExit(f"Need at least two arms; found {len(scores)}.")

    video_of = meta.loc[sample_ids, "video_id"].to_numpy()
    print(f"{len(scores)} arms, {len(sample_ids)} frames, "
          f"{np.unique(video_of).size} videos\n")
    print("arm mAP on the full split:")
    for arm in sorted(scores, key=lambda a: -mean_ap(targets, scores[a].mean(axis=0))):
        print(f"  {arm:<26}{mean_ap(targets, scores[arm].mean(axis=0)):.4f}  ({scores[arm].shape[0]} seeds)")

    rng = np.random.default_rng(args.seed)
    results: dict[str, Any] = {"sizes": {}}
    sizes = [n for n in args.subsample if n <= np.unique(video_of).size]
    if np.unique(video_of).size not in sizes:
        sizes.append(int(np.unique(video_of).size))

    print("\nseeds per arm: " + ", ".join(f"{n}={scores[n].shape[0]}" for n in sorted(scores)))
    print(f"\n{'videos':>8}{'seeds averaged':>16}{'seeds inside':>14}{'IQR (inside)':>22}{'resolved (inside)':>20}")
    for n in sorted(sizes):
        r_mean = resolution_at(scores, targets, video_of, n, args.n_boot, rng, "mean")
        r_in = resolution_at(scores, targets, video_of, n, args.n_boot, rng, "inside")
        results["sizes"][str(n)] = {"mean": r_mean, "inside": r_in}
        print(f"{n:>8}{r_mean['resolution_median']:>16.4f}{r_in['resolution_median']:>14.4f}"
              f"{r_in['resolution_q25']:>11.4f}–{r_in['resolution_q75']:<10.4f}"
              f"{r_in['pairs_resolved']:>8} of {r_in['n_pairs']}")

    r_seed = resolution_at(scores, targets, video_of, None, args.n_boot, rng, "seed_only")
    results["seed_only"] = r_seed
    print(f"\n  seed component alone (videos fixed): median half-width {r_seed['resolution_median']:.4f}")

    print("\n  The resolution is the median half-width of the paired difference")
    print("  interval across arm pairs: the smallest difference this benchmark")
    print("  returns reliably at that number of videos. 'Seeds inside' draws a")
    print("  seed per arm per replicate and is the figure to report; 'seeds")
    print("  averaged' is the video-sampling component alone.")
    print("\n  For reference, the four best published methods on Endoscapes span")
    print("  0.060 mAP, evaluated on 40 videos. The top three of thirteen SAGES")
    print("  submissions span 0.003, evaluated on 300.")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "benchmark_resolution.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwritten to {out / 'benchmark_resolution.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
