#!/usr/bin/env python3
"""Is the temporal gain above the resolution of the benchmark?

The resolution measured across encoder pairs on the official split is 0.042 mAP
at 300 videos. The temporal gain is 0.018. Read against that number the gain
sits below resolution and is not a measurement.

That reading would be wrong, and the reason is worth stating precisely. The 0.042
is the median half-width across pairs of *different encoders*: two models with
different weights, different features and different errors. The temporal
comparison is the same encoder, the same cached features, and the same probe
protocol, differing only in how many frames the head sees. Almost all of the
variance between those two arms is shared, and a paired bootstrap cancels it.

So the question is not whether 0.018 clears a threshold set elsewhere. It is
whether the paired interval on this specific difference excludes zero, and what
this comparison's own resolution is.

Both are computed here, from the saved per-window logits, resampled over videos
for the same reason the encoder comparison is: frames within a procedure share
patient, anatomy, camera and illumination.

**What the result licenses.** An interval excluding zero means the gain is real
at this test size, and the correct statement is that a comparison within an
encoder resolves differences an across-encoder comparison cannot. An interval
including zero means the temporal finding rests on a difference this benchmark
cannot return, and Section 5.17 needs restating.

Usage:
    python eval/temporal_resolution.py \\
        --probe-root ../outputs/temporal_probe \\
        --cache ../cache/dinov2_b/sages_official/test \\
        --manifest metadata/sages_frames_official_test.csv \\
        --output-dir ../outputs/resolution
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def read_index(cache: Path) -> list[str]:
    with open(cache / "index.csv", newline="", encoding="utf-8") as fh:
        return [r["sample_id"] for r in csv.DictReader(fh)]


def mean_ap(y: np.ndarray, p: np.ndarray) -> float:
    out = [average_precision_score(y[:, c], p[:, c])
           for c in range(y.shape[1]) if np.unique(y[:, c]).size > 1]
    return float(np.mean(out)) if out else float("nan")


def load_windows(root: Path, arm: str, prefix: str) -> dict[int, np.ndarray]:
    """Per-window scores, averaged over seeds.

    The temporal probe writes one logits file per window per seed. Seeds are
    averaged rather than compared, since head initialisation is not the factor
    under test.
    """
    out: dict[int, np.ndarray] = {}
    base = root / arm
    if not base.is_dir():
        raise SystemExit(f"{base} not found.")
    for window_dir in sorted(base.glob("k*")):
        try:
            k = int(window_dir.name.lstrip("k"))
        except ValueError:
            continue
        files = sorted(window_dir.glob(f"{prefix}_seed*.npz"))
        if not files:
            continue
        # Seeds are kept separate rather than averaged, so the bootstrap can
        # draw one per replicate. Averaging first treats head initialisation as
        # exact, and it is not: the same protocol moved by 0.012 between two
        # runs, against a video half-width of 0.015.
        out[k] = np.stack(
            [1.0 / (1.0 + np.exp(-np.load(f)["logits"])) for f in files]
        )
    return out


def paired_bootstrap(a: np.ndarray, b: np.ndarray, y: np.ndarray,
                     video_of: np.ndarray, n_boot: int,
                     rng: np.random.Generator,
                     seeds_inside: bool = True) -> dict[str, float]:
    """Interval on mAP(a) - mAP(b), resampled over videos.

    Both arms are scored on the same resampled videos in every replicate, so
    variance common to the two cancels. This is what makes a within-encoder
    comparison resolvable where an across-encoder one at the same test size is
    not.
    """
    videos = np.unique(video_of)
    rows = {v: np.flatnonzero(video_of == v) for v in videos}
    draws = np.empty(n_boot)
    unpaired_a = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.concatenate([rows[v] for v in
                              rng.choice(videos, videos.size, replace=True)])
        if a.ndim == 3 and seeds_inside:
            # One seed per arm per replicate, drawn independently: the seeds are
            # not paired in any meaningful sense, and pairing them would
            # understate the variance a reader should expect from a rerun.
            sa = a[rng.integers(a.shape[0])]
            sb = b[rng.integers(b.shape[0])]
        else:
            sa = a.mean(axis=0) if a.ndim == 3 else a
            sb = b.mean(axis=0) if b.ndim == 3 else b
        ap_a, ap_b = mean_ap(y[idx], sa[idx]), mean_ap(y[idx], sb[idx])
        draws[i] = ap_a - ap_b
        unpaired_a[i] = ap_a
    lo, hi = np.percentile(draws, 2.5), np.percentile(draws, 97.5)
    flat_a = a.mean(axis=0) if a.ndim == 3 else a
    flat_b = b.mean(axis=0) if b.ndim == 3 else b
    # The unpaired half-width is what these data would give if the two arms were
    # measured independently. Its ratio to the paired one says how much
    # per-video error the arms share: sqrt(2) times means none, and well below
    # that means most of the difficulty belongs to the videos and the labels
    # rather than to either model.
    single = float((np.percentile(unpaired_a, 97.5)
                    - np.percentile(unpaired_a, 2.5)) / 2)
    unpaired = float(np.sqrt(2) * single)
    paired_hw = float((hi - lo) / 2)
    return {
        "point": mean_ap(y, flat_a) - mean_ap(y, flat_b),
        "ci_low": float(lo), "ci_high": float(hi),
        "half_width": paired_hw,
        "unpaired_half_width": unpaired,
        "shared_fraction": float(1 - (paired_hw / unpaired) ** 2)
                           if unpaired > 0 else float("nan"),
        "excludes_zero": bool(lo > 0 or hi < 0),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--probe-root", required=True,
                   help="directory holding one subdirectory per arm")
    p.add_argument("--arms", nargs="+", default=["dinov2_b", "dinov3_b"])
    p.add_argument("--cache", required=True,
                   help="any cache with the same sample order as the probes")
    p.add_argument("--manifest", required=True)
    p.add_argument("--logits-prefix", default="test_logits_official")
    p.add_argument("--reference", type=int, default=1,
                   help="the window every other is compared against")
    p.add_argument("--seeds-outside", action="store_true",
                   help="average the probe seeds before resampling, as the "
                        "first version did. Kept only to reproduce the earlier "
                        "figures; the default propagates seed variance, which "
                        "at this scale is comparable to video variance.")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    ids = read_index(Path(args.cache))
    meta = pd.read_csv(args.manifest).set_index("sample_id").loc[ids]
    targets = meta[["c1_consensus", "c2_consensus", "c3_consensus"]].to_numpy(int)
    video_of = meta["video_id"].to_numpy()
    rng = np.random.default_rng(args.seed)

    print(f"{len(ids)} frames, {np.unique(video_of).size} videos")
    print(f"across-encoder resolution at this size: 0.042 (median half-width)\n")

    results: dict[str, Any] = {"arms": {}}
    for arm in args.arms:
        try:
            windows = load_windows(Path(args.probe_root), arm, args.logits_prefix)
        except SystemExit as exc:
            print(f"  {exc}"); continue
        if args.reference not in windows:
            print(f"  {arm}: no k={args.reference} to compare against"); continue

        print(f"{arm}")
        print(f"  {'k':>4}{'mAP':>9}{'vs k=%d' % args.reference:>11}"
              f"{'95% interval':>22}{'paired':>10}{'unpaired':>11}"
              f"{'shared':>9}{'res.':>7}")
        entry: dict[str, Any] = {}
        ref = windows[args.reference]
        for k in sorted(windows):
            # Averaged over seeds for the point estimate, which is the
            # figure that goes in a table. The interval below draws a seed per
            # replicate instead, so the two are answering different questions:
            # what the arm scores, and what a rerun would give.
            ap = mean_ap(targets, windows[k].mean(axis=0)
                         if windows[k].ndim == 3 else windows[k])
            if k == args.reference:
                print(f"  {k:>4}{ap:>9.4f}{'—':>11}{'':>22}{'':>12}{'':>10}")
                entry[str(k)] = {"map": ap}
                continue
            r = paired_bootstrap(windows[k], ref, targets, video_of,
                                 args.n_boot, rng,
                                 seeds_inside=not args.seeds_outside)
            entry[str(k)] = {"map": ap, **r}
            print(f"  {k:>4}{ap:>9.4f}{r['point']:>+11.4f}"
                  f"  [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]"
                  f"{r['half_width']:>10.4f}{r['unpaired_half_width']:>11.4f}"
                  f"{100*r['shared_fraction']:>8.0f}%"
                  f"{'yes' if r['excludes_zero'] else 'no':>7}")
        results["arms"][arm] = entry
        hw = [v["half_width"] for v in entry.values() if "half_width" in v]
        if hw:
            print(f"  median half-width within this encoder: {np.median(hw):.4f}")
        print()

    print("  A half-width well below 0.042 means the within-encoder comparison")
    print("  resolves what an across-encoder comparison at the same test size")
    print("  cannot, because the shared variance cancels. The resolution of a")
    print("  benchmark is a property of the comparison as much as of the split.")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "temporal_resolution.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwritten to {out / 'temporal_resolution.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
