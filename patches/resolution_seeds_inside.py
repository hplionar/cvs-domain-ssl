"""benchmark_resolution.py: seeds inside the bootstrap.

load_arm averaged sigmoid scores over seeds before resampling, so the interval
carried video-sampling variance only. This keeps per-seed arrays and lets each
replicate draw a seed per arm ("inside"), alongside the previous behaviour
("mean") and a seed-only decomposition. Adds MDE at 80% power per pair.
"""
from pathlib import Path
p = Path("eval/benchmark_resolution.py"); src = p.read_text()
def rep(old, new):
    global src
    n = src.count(old); assert n == 1, f"expected 1 match, found {n}:\n{old[:90]}"
    src = src.replace(old, new)

rep('''def load_arm(probe_dir: Path, prefix: str) -> np.ndarray | None:
    """Mean sigmoid score over seeds, [N, 3].

    Averaged over seeds rather than treated as separate arms: seed variation is
    a property of the head, and the question here is what the benchmark can
    resolve between encoders.
    """
    files = sorted(probe_dir.glob(f"{prefix}_seed*.npz"))
    if not files:
        return None
    scores = []
    for path in files:
        payload = np.load(path)
        scores.append(1.0 / (1.0 + np.exp(-payload["logits"])))
    return np.mean(scores, axis=0)
''',
'''def load_arm(probe_dir: Path, prefix: str) -> np.ndarray | None:
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
''')

rep('''def resolution_at(
    scores: dict[str, np.ndarray],
    targets: np.ndarray,
    video_of: np.ndarray,
    n_videos: int | None,
    n_boot: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Median half-width of the paired difference interval, over all arm pairs.

    With `n_videos`, each replicate first draws that many videos without
    replacement and then resamples them with replacement, which estimates the
    resolution a benchmark of that size would have. Without it, all videos are
    resampled, which estimates the resolution of the benchmark as released.
    """
    videos = np.unique(video_of)
    rows = {v: np.flatnonzero(video_of == v) for v in videos}
    names = sorted(scores)

    widths: dict[tuple[str, str], list[float]] = {p: [] for p in combinations(names, 2)}
    for _ in range(n_boot):
        pool = (rng.choice(videos, n_videos, replace=False)
                if n_videos and n_videos < videos.size else videos)
        draw = rng.choice(pool, pool.size, replace=True)
        idx = np.concatenate([rows[v] for v in draw])
        y = targets[idx]
        aps = {n: mean_ap(y, scores[n][idx]) for n in names}
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
        }
''',
'''def resolution_at(
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
''')

rep('''    hw = np.array([v["half_width"] for v in half.values()])
    return {
        "n_videos": int(n_videos or videos.size),
        "n_pairs": len(half),
''',
'''    hw = np.array([v["half_width"] for v in half.values()])
    return {
        "n_videos": int(n_videos or videos.size),
        "seed_mode": seed_mode,
        "seeds_per_arm": {n: int(scores[n].shape[0]) for n in names},
        "n_pairs": len(half),
''')

rep('''    print(f"\\n{'videos':>8}{'resolution':>13}{'IQR':>22}{'pairs resolved':>18}")
    for n in sorted(sizes):
        r = resolution_at(scores, targets, video_of, n, args.n_boot, rng)
        results["sizes"][str(n)] = r
        print(f"{n:>8}{r['resolution_median']:>13.4f}"
              f"{r['resolution_q25']:>11.4f}–{r['resolution_q75']:<10.4f}"
              f"{r['pairs_resolved']:>8} of {r['n_pairs']}")

    print("\\n  The resolution is the median half-width of the paired difference")
    print("  interval across arm pairs: the smallest difference this benchmark")
    print("  returns reliably at that number of videos.")
''',
'''    print("\\nseeds per arm: " + ", ".join(f"{n}={scores[n].shape[0]}" for n in sorted(scores)))
    print(f"\\n{'videos':>8}{'seeds averaged':>16}{'seeds inside':>14}{'IQR (inside)':>22}{'resolved (inside)':>20}")
    for n in sorted(sizes):
        r_mean = resolution_at(scores, targets, video_of, n, args.n_boot, rng, "mean")
        r_in = resolution_at(scores, targets, video_of, n, args.n_boot, rng, "inside")
        results["sizes"][str(n)] = {"mean": r_mean, "inside": r_in}
        print(f"{n:>8}{r_mean['resolution_median']:>16.4f}{r_in['resolution_median']:>14.4f}"
              f"{r_in['resolution_q25']:>11.4f}–{r_in['resolution_q75']:<10.4f}"
              f"{r_in['pairs_resolved']:>8} of {r_in['n_pairs']}")

    r_seed = resolution_at(scores, targets, video_of, None, args.n_boot, rng, "seed_only")
    results["seed_only"] = r_seed
    print(f"\\n  seed component alone (videos fixed): median half-width {r_seed['resolution_median']:.4f}")

    print("\\n  The resolution is the median half-width of the paired difference")
    print("  interval across arm pairs: the smallest difference this benchmark")
    print("  returns reliably at that number of videos. 'Seeds inside' draws a")
    print("  seed per arm per replicate and is the figure to report; 'seeds")
    print("  averaged' is the video-sampling component alone.")
''')

p.write_text(src); print("patched", p)
