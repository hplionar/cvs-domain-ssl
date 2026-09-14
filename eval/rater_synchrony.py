#!/usr/bin/env python3
"""Do the raters' label boundaries coincide?

The falsifier for the episode account. If a label marks a *view episode* -- a
stretch in which an exposure is held, ended when the camera moves or an
instrument crosses the field -- then the thing that ends it is in the video, and
all three raters should end their runs at the same frame. If instead each rater
dithers near their own bar on a continuously held view, boundaries fall
independently.

Two measurements, both from labels alone.

**Transition coincidence.** For each rater, the frames where their judgement
changes. Then the fraction of those transitions with another rater's transition
within a tolerance. Compared against a null that circularly shifts one rater's
sequence within its video, which preserves run lengths and positive rate exactly
and destroys only the alignment. A shift null is used rather than a permutation
because permuting would also destroy the run structure, and run structure alone
produces coincidence when runs are short.

**Nesting frame by frame.** On a criterion where the rates are nested in
aggregate, a threshold account additionally requires nesting *at each frame*:
the strict rater's positive frames should be a subset of the lenient rater's, not
merely fewer. Reported per ordered pair, since aggregate nesting can hold while
the frames themselves interleave.

Reading the result:

    coincidence far above null   the episodes are in the video. What ends a run
                                 is visible, and the account stands.
    coincidence at null          boundaries are independent. Flicker is reading
                                 noise near a bar, the blank-frame inheritance
                                 was annotation-tool inertia, and the ordinary
                                 account -- a frame-level truth reported with
                                 error -- takes most of the ground.

Usage:
    python eval/rater_synchrony.py \\
        --metadata ../datasets/endoscapes/all_metadata.csv \\
        --output-dir ../outputs/threshold_tests
"""

from __future__ import annotations

import argparse
import ast
import json
from itertools import combinations, permutations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CRITERIA = ("C1", "C2", "C3")
RATERS = (1, 2, 3)


def per_rater_matrix(df: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    cols = [f"cvs_annotator_{r}" for r in RATERS]
    keep = df[cols].notna().all(axis=1)
    df = df[keep].sort_values(["vid", "frame"]).reset_index(drop=True)
    out = np.zeros((len(df), len(RATERS), len(CRITERIA)), dtype=int)
    for i, r in enumerate(RATERS):
        for j, value in enumerate(df[f"cvs_annotator_{r}"].apply(ast.literal_eval)):
            out[j, i] = value
    return out, df


def transitions(y: np.ndarray) -> np.ndarray:
    """Indices where the judgement changes from the previous frame.

    A transition is attributed to the later of the two frames, so a run that
    begins at index i has a transition at i. Both directions are counted: a view
    being lost ends a positive run and is as much an event as one beginning.
    """
    if y.size < 2:
        return np.array([], dtype=int)
    return np.flatnonzero(np.diff(y) != 0) + 1


def coincidence(a: np.ndarray, b: np.ndarray, tol: int) -> tuple[int, int]:
    """How many of a's transitions have one of b's within tol frames."""
    if a.size == 0:
        return 0, 0
    if b.size == 0:
        return 0, int(a.size)
    hits = sum(1 for t in a if np.min(np.abs(b - t)) <= tol)
    return hits, int(a.size)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metadata", required=True)
    p.add_argument("--tolerance", type=int, default=1,
                   help="frames within which two transitions count as coincident. "
                        "One frame is 5 seconds at this sampling rate.")
    p.add_argument("--n-null", type=int, default=200,
                   help="circular shifts per video for the null")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    df = pd.read_csv(args.metadata)
    df = df[df.cvs_annotator_1.notna()]
    votes, df = per_rater_matrix(df)
    groups = {v: g.index.to_numpy() for v, g in df.groupby("vid")}
    rng = np.random.default_rng(args.seed)

    print(f"{len(df)} frames, {len(groups)} videos, tolerance "
          f"{args.tolerance} frame(s) = {5*args.tolerance}s\n")

    results: dict[str, Any] = {"tolerance": args.tolerance, "criteria": {}}

    for j, criterion in enumerate(CRITERIA):
        print("=" * 70)
        print(f"{criterion}: do transitions coincide?")
        print("=" * 70)
        entry: dict[str, Any] = {"pairs": {}}

        print(f"  {'pair':<8}{'observed':>10}{'null':>9}{'ratio':>8}"
              f"{'a-transitions':>15}")
        for a, b in combinations(RATERS, 2):
            hits = total = 0
            null_hits = np.zeros(args.n_null)
            for rows in groups.values():
                ya = votes[rows, RATERS.index(a), j]
                yb = votes[rows, RATERS.index(b), j]
                ta, tb = transitions(ya), transitions(yb)
                h, n = coincidence(ta, tb, args.tolerance)
                hits += h; total += n
                if ta.size == 0 or yb.size < 2:
                    continue
                # Circular shift preserves run lengths and rate, breaks alignment.
                for s in range(args.n_null):
                    shift = int(rng.integers(1, max(yb.size, 2)))
                    tb_null = transitions(np.roll(yb, shift))
                    null_hits[s] += coincidence(ta, tb_null, args.tolerance)[0]

            obs = hits / max(total, 1)
            null = float(null_hits.mean()) / max(total, 1)
            entry["pairs"][f"{a}-{b}"] = {
                "observed": obs, "null_mean": null,
                "null_sd": float(null_hits.std()) / max(total, 1),
                "ratio": obs / null if null > 0 else float("nan"),
                "n_transitions": total,
            }
            print(f"  {a}-{b:<6}{100*obs:>9.1f}%{100*null:>8.1f}%"
                  f"{obs/null if null > 0 else float('nan'):>8.2f}{total:>15}")

        # Frame-level nesting, which aggregate rate nesting does not imply.
        rates = {r: float(votes[:, RATERS.index(r), j].mean()) for r in RATERS}
        entry["rates"] = rates
        entry["frame_nesting"] = {}
        print(f"\n  frame-level nesting (strict positives inside lenient's)")
        print(f"  {'pair':<10}{'strict pos':>12}{'also lenient':>14}{'share':>9}")
        for a, b in permutations(RATERS, 2):
            if rates[a] >= rates[b]:
                continue
            ya = votes[:, RATERS.index(a), j]
            yb = votes[:, RATERS.index(b), j]
            inside = int(((ya == 1) & (yb == 1)).sum())
            entry["frame_nesting"][f"{a}<{b}"] = {
                "strict_positives": int(ya.sum()), "also_lenient": inside,
                "share": inside / max(int(ya.sum()), 1),
            }
            print(f"  {a} < {b:<6}{int(ya.sum()):>12}{inside:>14}"
                  f"{100*inside/max(int(ya.sum()),1):>8.1f}%")
        results["criteria"][criterion] = entry
        print()

    print("  A ratio near one means boundaries fall where a shifted sequence's")
    print("  would: independent, and the runs are not tracking anything in the")
    print("  video. Well above one means the raters are ending runs together,")
    print("  which is what a lost view would produce.")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "rater_synchrony.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwritten to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
