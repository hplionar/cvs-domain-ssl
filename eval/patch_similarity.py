#!/usr/bin/env python3
"""Within-frame patch similarity, before and after continued pretraining.

Continued pretraining of DINOv2 on SAGES raised the entropy of the mean head's
per-patch contributions by 0.53 nats on every criterion and cut the mass held by
the ten highest patches from 0.38-0.42 to 0.17-0.29. The decision became
spatially diffuse.

The simplest explanation is mechanical rather than semantic. A linear head over
mean-pooled tokens gives patch i the contribution (1/P)(w . h_i). If adaptation
made the patch tokens within a frame more similar to one another, then every
w . h_i moves toward the same value, contributions flatten, and entropy rises --
with no change in what any individual patch represents. Testing that requires
looking at the tokens, not the head.

Three quantities, none of which involves the head or any alignment between the
two feature spaces, so both arms can be measured on their own terms:

    within-frame cosine   mean pairwise cosine between the patch tokens of one
                          frame. High values mean the tokens carry little
                          spatial differentiation.
    effective rank        exp of the entropy of the normalised eigenvalue
                          spectrum of the frame's token covariance. Counts how
                          many directions the tokens actually occupy, out of the
                          feature dimension. Falls when tokens collapse toward a
                          common direction.
    token norm spread     coefficient of variation of ||h_i|| within a frame,
                          which separates two ways of becoming uniform:
                          tokens pointing the same way, and tokens of the same
                          magnitude.

The CLS token is excluded. It is a global summary by construction and would
raise the apparent similarity for reasons unrelated to the patch grid.

Interpretation. If the adapted arm shows higher within-frame cosine and lower
effective rank, uniformly across frames, the entropy result is explained by a
loss of spatial differentiation in the encoder and not by a change in what the
head attends to. This would be a partial collapse of *spatial* structure, which
the collapse detector in ``pretrain_dino.py`` does not watch for: that monitors
the rank of the teacher's pooled CLS output across a batch, which stays healthy
even if the patch tokens within each frame become interchangeable.

Usage:
    python eval/patch_similarity.py \\
        --arm base=../cache/dinov2_b/sages/val \\
        --arm adapted=../cache/dinov2_b_adapted/sages/val \\
        --manifest metadata/sages_frames_internal_split.csv \\
        --output-dir ../outputs/patch_similarity
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CRITERIA = ("c1", "c2", "c3")
RATERS = (1, 2, 3)
BLOCK_BYTES = 1 << 30


def read_index(cache_dir: Path) -> list[str]:
    with open(cache_dir / "index.csv", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if [int(r["row"]) for r in rows] != list(range(len(rows))):
        raise ValueError(f"{cache_dir/'index.csv'} rows are not 0..N-1 in order.")
    return [r["sample_id"] for r in rows]


def frame_statistics(cache_dir: Path) -> dict[str, np.ndarray]:
    """Per-frame token statistics, computed in blocks.

    Returns arrays of length N: mean pairwise cosine, effective rank, and the
    coefficient of variation of the token norms.
    """
    tokens = np.load(cache_dir / "tokens.npy", mmap_mode="r")
    n, n_patches, dim = tokens.shape
    block = max(1, int(BLOCK_BYTES // max(n_patches * dim * 4, 1)))

    cosines = np.empty(n, dtype=np.float32)
    ranks = np.empty(n, dtype=np.float32)
    norm_cv = np.empty(n, dtype=np.float32)

    for start in range(0, n, block):
        stop = min(start + block, n)
        h = np.asarray(tokens[start:stop], dtype=np.float32)          # [B, P, D]

        norms = np.linalg.norm(h, axis=2)                              # [B, P]
        unit = h / (norms[:, :, None] + 1e-8)

        # Mean pairwise cosine over the P(P-1)/2 distinct pairs. Computed from
        # the Gram matrix rather than by enumeration: sum of all entries minus
        # the P diagonal ones, halved.
        gram_sum = np.einsum("bpd,bqd->b", unit, unit)
        cosines[start:stop] = (gram_sum - n_patches) / (n_patches * (n_patches - 1))

        # Effective rank of the token covariance within the frame. Centred,
        # because the mean token is exactly what the pooling discards, and an
        # uncentred spectrum would be dominated by it.
        centred = h - h.mean(axis=1, keepdims=True)
        # Gram is P x P, which is smaller than D x D here and has the same
        # non-zero spectrum.
        g = np.einsum("bpd,bqd->bpq", centred, centred)
        eig = np.linalg.eigvalsh(g)
        eig = np.clip(eig, 0.0, None)
        p_spec = eig / (eig.sum(axis=1, keepdims=True) + 1e-12)
        entropy = -(p_spec * np.log(p_spec + 1e-12)).sum(axis=1)
        ranks[start:stop] = np.exp(entropy)

        norm_cv[start:stop] = norms.std(axis=1) / (norms.mean(axis=1) + 1e-8)

    return {"cosine": cosines, "effective_rank": ranks, "norm_cv": norm_cv,
            "n_patches": n_patches, "dim": dim}


def summarise(x: np.ndarray) -> dict[str, float]:
    return {"mean": float(x.mean()),
            "sd": float(x.std(ddof=1)) if x.size > 1 else 0.0,
            "p10": float(np.percentile(x, 10)),
            "p90": float(np.percentile(x, 90))}


def parse_arm(spec: str) -> tuple[str, Path]:
    name, path = spec.split("=", 1)
    return name, Path(path)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", action="append", required=True, type=parse_arm,
                   help="name=cache_dir (repeatable; two or more)")
    p.add_argument("--manifest", required=True)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    meta = pd.read_csv(args.manifest).set_index("sample_id")

    stats, ids_ref = {}, None
    for name, cache in args.arm:
        ids = read_index(cache)
        if ids_ref is None:
            ids_ref = ids
        elif ids != ids_ref:
            raise SystemExit(
                f"{name} holds different samples in a different order from the "
                f"first arm. The comparison requires identical frames."
            )
        print(f"reading {name}  ({cache})")
        stats[name] = frame_statistics(cache)

    assert ids_ref is not None
    df = meta.loc[ids_ref].reset_index()

    results: dict[str, Any] = {"n_frames": len(ids_ref), "arms": {}}

    print(f"\nWithin-frame token statistics, mean over frames")
    print(f"{'arm':<12}{'patches':>9}{'dim':>6}{'cosine':>10}{'eff.rank':>10}"
          f"{'% of P':>9}{'norm CV':>10}")
    for name, s in stats.items():
        summary = {k: summarise(s[k]) for k in ("cosine", "effective_rank", "norm_cv")}
        results["arms"][name] = {"n_patches": int(s["n_patches"]), "dim": int(s["dim"]),
                                 **summary}
        print(f"{name:<12}{s['n_patches']:>9}{s['dim']:>6}"
              f"{summary['cosine']['mean']:>10.4f}"
              f"{summary['effective_rank']['mean']:>10.2f}"
              f"{100*summary['effective_rank']['mean']/s['n_patches']:>8.1f}%"
              f"{summary['norm_cv']['mean']:>10.4f}")

    names = list(stats)
    if len(names) == 2:
        a, b = names
        print(f"\nChange, {a} -> {b}")
        for key, label in (("cosine", "within-frame cosine"),
                           ("effective_rank", "effective rank"),
                           ("norm_cv", "norm CV")):
            x, y = stats[a][key], stats[b][key]
            # Paired: the same frames in the same order, so the difference is
            # taken per frame rather than between two marginal means.
            d = y - x
            share = float((d > 0).mean())
            print(f"  {label:<22}{x.mean():>9.4f} -> {y.mean():>8.4f}"
                  f"   change {d.mean():>+8.4f}"
                  f"   ({100*share:.0f}% of frames increased)")
            results.setdefault("paired_change", {})[key] = {
                "from": float(x.mean()), "to": float(y.mean()),
                "mean_change": float(d.mean()),
                "sd_change": float(d.std(ddof=1)) if d.size > 1 else 0.0,
                "fraction_increased": share,
            }

        print(f"\n  A rise in within-frame cosine with a fall in effective rank means")
        print(f"  the patch tokens became more alike, which flattens any linear")
        print(f"  head's contribution map without any change in what a patch means.")

        # Stratified, since the arms' downstream difference was stratum-specific.
        votes = {c: df[[f"{c}_rater{r}" for r in RATERS]].to_numpy(int).sum(axis=1)
                 for c in CRITERIA}
        print(f"\nChange in within-frame cosine by annotator-agreement stratum")
        print(f"{'crit':<5}{'unanimous':>12}{'contested':>12}")
        results["by_stratum"] = {}
        for c in CRITERIA:
            un = (votes[c] == 0) | (votes[c] == 3)
            d = stats[b]["cosine"] - stats[a]["cosine"]
            row = {"unanimous": float(d[un].mean()), "contested": float(d[~un].mean())}
            results["by_stratum"][c] = row
            print(f"{c:<5}{row['unanimous']:>+12.4f}{row['contested']:>+12.4f}")
        print("  The strata partition the same frames three ways, so these differ")
        print("  between criteria only through which frames were contested.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "patch_similarity.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwritten to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
