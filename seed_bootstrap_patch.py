#!/usr/bin/env python3
"""Put seeds inside the bootstrap, and measure the unpaired half-width.

Two changes to eval/temporal_resolution.py and eval/benchmark_resolution.py,
both prompted by the same doubt.

**Seeds inside.** Both scripts average the three probe seeds and then resample
videos, which treats head initialisation as though it were exact. It is not: the
temporal figures moved by 0.012 between two runs of the same protocol with
different seeds, against a video half-width of 0.015. At that scale seed variance
rivals video variance, and an interval that excludes it is too narrow. Drawing a
seed per arm per replicate propagates both sources, and the resulting half-width
is the honest one. If it grows from 0.015 to something near 0.025, the finest
tick is seed-limited rather than benchmark-limited and the three-tick argument
becomes a two-tick one.

**Unpaired alongside paired.** The paired half-width is small when two arms share
per-video error. How much they share is itself the quantity of interest: if the
paired interval sits well below sqrt(2) times the single-arm interval, unrelated
encoders share most of their per-video error, and the difficulty belongs to the
videos and the labels rather than to any model. If it sits at sqrt(2) times,
there is no shared component and the pairing buys nothing beyond what
independence would give. Reporting the ratio makes the claim testable rather
than asserted.

Run from the repository root:
    python seed_bootstrap_patch.py
"""

from __future__ import annotations

import pathlib


def patch(path: str, pairs: list[tuple[str, str, str]]) -> None:
    p = pathlib.Path(path)
    if not p.is_file():
        print(f"  MISSING  {path}")
        return
    s = p.read_text()
    for old, new, label in pairs:
        if new.strip().splitlines()[0].strip() in s:
            print(f"  skip     {path}: {label} (already applied)")
            continue
        if old not in s:
            print(f"  NO MATCH {path}: {label}")
            continue
        s = s.replace(old, new, 1)
        print(f"  applied  {path}: {label}")
    p.write_text(s)


def main() -> int:
    # --- temporal_resolution.py ------------------------------------------
    patch("eval/temporal_resolution.py", [
        (
            '''    out: dict[int, np.ndarray] = {}
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
        out[k] = np.mean(
            [1.0 / (1.0 + np.exp(-np.load(f)["logits"])) for f in files], axis=0
        )
    return out''',
            '''    out: dict[int, np.ndarray] = {}
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
    return out''',
            "seeds kept separate",
        ),
        (
            '''def paired_bootstrap(a: np.ndarray, b: np.ndarray, y: np.ndarray,
                     video_of: np.ndarray, n_boot: int,
                     rng: np.random.Generator) -> dict[str, float]:''',
            '''def paired_bootstrap(a: np.ndarray, b: np.ndarray, y: np.ndarray,
                     video_of: np.ndarray, n_boot: int,
                     rng: np.random.Generator,
                     seeds_inside: bool = True) -> dict[str, float]:''',
            "seeds_inside parameter",
        ),
        (
            '''    videos = np.unique(video_of)
    rows = {v: np.flatnonzero(video_of == v) for v in videos}
    draws = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.concatenate([rows[v] for v in
                              rng.choice(videos, videos.size, replace=True)])
        draws[i] = mean_ap(y[idx], a[idx]) - mean_ap(y[idx], b[idx])''',
            '''    videos = np.unique(video_of)
    rows = {v: np.flatnonzero(video_of == v) for v in videos}
    n_seeds = a.shape[0] if a.ndim == 3 else 1
    draws = np.empty(n_boot)
    unpaired_a = np.empty(n_boot)
    unpaired_b = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.concatenate([rows[v] for v in
                              rng.choice(videos, videos.size, replace=True)])
        if a.ndim == 3:
            # One seed drawn per arm per replicate. Drawing independently for
            # the two arms is deliberate: the seeds are not paired in any
            # meaningful sense, and pairing them would understate the variance
            # a reader should expect from a rerun.
            sa = a[rng.integers(n_seeds)] if seeds_inside else a.mean(axis=0)
            sb = b[rng.integers(b.shape[0])] if seeds_inside else b.mean(axis=0)
        else:
            sa, sb = a, b
        ap_a, ap_b = mean_ap(y[idx], sa[idx]), mean_ap(y[idx], sb[idx])
        draws[i] = ap_a - ap_b
        unpaired_a[i], unpaired_b[i] = ap_a, ap_b''',
            "seed draw inside the replicate",
        ),
        (
            '''    lo, hi = np.percentile(draws, 2.5), np.percentile(draws, 97.5)
    return {
        "point": mean_ap(y, a) - mean_ap(y, b),
        "ci_low": float(lo), "ci_high": float(hi),
        "half_width": float((hi - lo) / 2),
        "excludes_zero": bool(lo > 0 or hi < 0),
    }''',
            '''    lo, hi = np.percentile(draws, 2.5), np.percentile(draws, 97.5)
    flat_a = a.mean(axis=0) if a.ndim == 3 else a
    flat_b = b.mean(axis=0) if b.ndim == 3 else b

    # The unpaired half-width is what the same data would give if the two arms
    # were measured independently. Its ratio to the paired one says how much
    # per-video error the arms share: sqrt(2) means none, and well below sqrt(2)
    # means most of the difficulty belongs to the videos rather than the models.
    single = float((np.percentile(unpaired_a, 97.5)
                    - np.percentile(unpaired_a, 2.5)) / 2)
    paired_hw = float((hi - lo) / 2)
    return {
        "point": mean_ap(y, flat_a) - mean_ap(y, flat_b),
        "ci_low": float(lo), "ci_high": float(hi),
        "half_width": paired_hw,
        "single_arm_half_width": single,
        "unpaired_half_width": float(np.sqrt(2) * single),
        "shared_fraction": float(1 - (paired_hw / (np.sqrt(2) * single)) ** 2)
                           if single > 0 else float("nan"),
        "excludes_zero": bool(lo > 0 or hi < 0),
    }''',
            "unpaired half-width and shared fraction",
        ),
        (
            '''            print(f"  {k:>4}{ap:>9.4f}{r['point']:>+11.4f}"
                  f"  [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]"
                  f"{r['half_width']:>12.4f}{'yes' if r['excludes_zero'] else 'no':>10}")''',
            '''            print(f"  {k:>4}{ap:>9.4f}{r['point']:>+11.4f}"
                  f"  [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]"
                  f"{r['half_width']:>12.4f}{r['unpaired_half_width']:>11.4f}"
                  f"{100*r['shared_fraction']:>9.0f}%"
                  f"{'yes' if r['excludes_zero'] else 'no':>9}")''',
            "report the unpaired column",
        ),
        (
            '''        print(f"  {'k':>4}{'mAP':>9}{'vs k=%d' % args.reference:>11}"
              f"{'95% interval':>22}{'half-width':>12}{'resolved':>10}")''',
            '''        print(f"  {'k':>4}{'mAP':>9}{'vs k=%d' % args.reference:>11}"
              f"{'95% interval':>22}{'paired':>12}{'unpaired':>11}"
              f"{'shared':>9}{'res.':>9}")''',
            "header",
        ),
        (
            '''    p.add_argument("--n-boot", type=int, default=2000)''',
            '''    p.add_argument("--seeds-outside", action="store_true",
                   help="average the probe seeds before resampling, as the "
                        "first version did. Kept only to reproduce the earlier "
                        "figures; the default propagates seed variance, which "
                        "at this scale is comparable to video variance.")
    p.add_argument("--n-boot", type=int, default=2000)''',
            "seeds-outside flag",
        ),
        (
            '''            r = paired_bootstrap(windows[k], ref, targets, video_of,
                                 args.n_boot, rng)''',
            '''            r = paired_bootstrap(windows[k], ref, targets, video_of,
                                 args.n_boot, rng,
                                 seeds_inside=not args.seeds_outside)''',
            "pass the flag",
        ),
    ])

    print()
    print("Now run, and compare against the previous 0.0150 for DINOv2:")
    print()
    print("  python eval/temporal_resolution.py \\")
    print("    --probe-root ../outputs/temporal_probe \\")
    print("    --cache ../cache/dinov2_b/sages_official/test \\")
    print("    --manifest metadata/sages_frames_official_test.csv \\")
    print("    --output-dir ../outputs/resolution")
    print()
    print("If the paired half-width grows from 0.015 to near 0.025, the finest")
    print("tick is seed-limited rather than benchmark-limited, and the")
    print("three-tick claim becomes a two-tick one. Run --seeds-outside to")
    print("confirm the difference is the seeds and not something else.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
