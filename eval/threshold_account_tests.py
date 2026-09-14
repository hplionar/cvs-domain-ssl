#!/usr/bin/env python3
"""Three tests that separate a rater-threshold account from independent noise.

The account under test: the label is a judgement about a view episode -- a few
seconds in which an exposure is held -- made by a surgeon applying their own
criterion-specific bar. The frame is a 0.2 Hz sample of that episode, and
disagreement is the band between the strictest and most lenient bar rather than
error around a common truth.

The competing account: a frame-level truth exists and raters report it with
independent, roughly zero-mean error.

Three consequences distinguish them, and all three are countable from the
released annotations with no model and no compute.

**1. Disagreement should be nested.** If raters differ only in where they set a
bar, then whenever the strict rater says yes the lenient one should also say
yes. Strict-yes-with-lenient-no should be near-absent. Independent noise
produces both directions in proportion to the rates involved. Reported against
the rate expected if the two raters were independent, since some nesting arises
by chance when one rate is much lower than the other.

**2. Isolated positives should be a sampling artefact, not flicker.** A criterion
satisfied for one 5-second sample and not the next is implausible as a property
of the dissection. Under the episode account, isolated positives arise either
from episodes shorter than the 10 seconds needed to catch two samples, or from
the ragged edge of a longer episode where only the middle rater's bar is
crossed. The first is visible per rater; the second only in the majority. So the
isolated fraction should be **lower per rater than in the majority vote** -- the
vote manufactures flicker the individual judgements do not have.

**3. Blank frames should inherit their episode's label.** A frame with no image
content cannot be judged from the image. Under the episode account a positive
blank frame sits inside or adjacent to a positive stretch and a negative one
sits before dissection begins. Under noise they fall uniformly. Reported as the
label of the nearest annotated non-blank neighbour and the position of the blank
frame within the video.

None of these is decisive alone. The first is the cheapest thing that could
falsify the account outright: disagreement that is not nested is not a bar.

Usage:
    python eval/threshold_account_tests.py \\
        --metadata ../datasets/endoscapes/all_metadata.csv \\
        --manifest metadata/endoscapes_frames.csv \\
        --dataset-root ../datasets/endoscapes \\
        --output-dir ../outputs/threshold_tests
"""

from __future__ import annotations

import argparse
import ast
import json
from itertools import permutations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CRITERIA = ("C1", "C2", "C3")
RATERS = (1, 2, 3)


def per_rater_matrix(df: pd.DataFrame) -> np.ndarray:
    """[N, 3 raters, 3 criteria] of binary judgements.

    The columns hold a stringified list of three values, so they are parsed
    rather than read as numbers. Rows with any rater missing are dropped, since
    a partial row cannot contribute to a pairwise comparison and silently
    treating a missing judgement as negative would manufacture nesting.
    """
    cols = [f"cvs_annotator_{r}" for r in RATERS]
    have = df[cols].notna().all(axis=1)
    if not have.all():
        print(f"  dropped {int((~have).sum())} rows with a missing rater")
    df = df[have]
    out = np.zeros((len(df), len(RATERS), len(CRITERIA)), dtype=int)
    for i, r in enumerate(RATERS):
        parsed = df[f"cvs_annotator_{r}"].apply(ast.literal_eval)
        for j, value in enumerate(parsed):
            out[j, i] = value
    return out, df.reset_index(drop=True)


def test_nesting(votes: np.ndarray) -> dict[str, Any]:
    """Is disagreement nested, as a difference in threshold implies?

    For every ordered pair of raters (a, b) where a is the stricter on this
    criterion, count frames where a said yes and b said no. Under a pure
    threshold account that count is zero. Compared against the count expected if
    the two were independent at their observed rates, because when one rate is
    3.7% and the other 30% a good deal of nesting happens by chance.
    """
    result: dict[str, Any] = {}
    for j, criterion in enumerate(CRITERIA):
        rates = {r: float(votes[:, i, j].mean()) for i, r in enumerate(RATERS)}
        order = sorted(RATERS, key=lambda r: rates[r])  # strictest first
        entry = {"rates": rates, "strict_to_lenient": order, "pairs": {}}
        n = len(votes)
        for a, b in permutations(RATERS, 2):
            if rates[a] >= rates[b]:
                continue  # only stricter-against-more-lenient
            ya = votes[:, RATERS.index(a), j]
            yb = votes[:, RATERS.index(b), j]
            violations = int(((ya == 1) & (yb == 0)).sum())
            # Expected under independence at the same marginal rates.
            expected = rates[a] * (1 - rates[b]) * n
            entry["pairs"][f"{a}>{b}"] = {
                "strict_yes_lenient_no": violations,
                "expected_if_independent": expected,
                "ratio": violations / expected if expected > 0 else float("nan"),
                "strict_positives": int(ya.sum()),
                "violation_share_of_strict_positives":
                    violations / max(int(ya.sum()), 1),
            }
        result[criterion] = entry
    return result


def runs(y: np.ndarray) -> list[int]:
    """Lengths of consecutive positive stretches."""
    out, n = [], 0
    for x in y:
        if x:
            n += 1
        elif n:
            out.append(n); n = 0
    if n:
        out.append(n)
    return out


def test_isolation(votes: np.ndarray, df: pd.DataFrame) -> dict[str, Any]:
    """Are isolated positives rarer per rater than in the majority vote?

    Under the episode account the majority vote manufactures flicker: at the
    ragged edge of an episode only the middle rater's bar is crossed, so the
    vote turns positive for one sample where no individual rater's judgement
    does. Under independent noise the majority is the smoother signal, not the
    rougher one, because averaging three noisy reports reduces variance.
    """
    ordered = df.sort_values(["vid", "frame"])
    idx = ordered.index.to_numpy()
    result: dict[str, Any] = {}
    for j, criterion in enumerate(CRITERIA):
        entry: dict[str, Any] = {}
        for i, r in enumerate(RATERS):
            lengths = []
            for _, g in ordered.groupby("vid"):
                lengths += runs(votes[g.index.to_numpy(), i, j])
            entry[f"rater_{r}"] = {
                "n_runs": len(lengths),
                "isolated": int(sum(1 for x in lengths if x == 1)),
                "isolated_share": (sum(1 for x in lengths if x == 1) / len(lengths)
                                   if lengths else float("nan")),
                "median_run": float(np.median(lengths)) if lengths else float("nan"),
            }
        lengths = []
        for _, g in ordered.groupby("vid"):
            rows = g.index.to_numpy()
            lengths += runs((votes[rows, :, j].sum(axis=1) >= 2).astype(int))
        entry["majority"] = {
            "n_runs": len(lengths),
            "isolated": int(sum(1 for x in lengths if x == 1)),
            "isolated_share": (sum(1 for x in lengths if x == 1) / len(lengths)
                               if lengths else float("nan")),
            "median_run": float(np.median(lengths)) if lengths else float("nan"),
        }
        result[criterion] = entry
    return result


def test_blank_frames(votes: np.ndarray, df: pd.DataFrame, root: Path,
                      manifest: pd.DataFrame) -> dict[str, Any]:
    """Do blank frames carry the label of the stretch they sit in?

    A frame with no image content cannot be judged from the image, so its label
    came from somewhere else. Under the episode account a positive blank frame
    sits inside or beside a positive stretch and a negative one sits before the
    dissection has progressed. Reported as the nearest annotated non-blank
    neighbour's label and the blank frame's relative position in its video.
    """
    from PIL import Image

    lookup = manifest.set_index("sample_id")
    df = df.copy()
    df["sample_id"] = df.vid.astype(str) + "_" + df.frame.astype(str)
    have = df.sample_id.isin(lookup.index)
    paths = df.loc[have, "sample_id"].map(lookup.relative_path)

    lum = {}
    for sample, rel in paths.items():
        p = root / str(rel)
        if p.is_file():
            lum[df.loc[sample, "sample_id"]] = float(
                np.asarray(Image.open(p).convert("L").resize((32, 32))).mean()
            )
    df["luminance"] = df.sample_id.map(lum)
    blank = df.luminance < 5

    if not blank.any():
        return {"n_blank": 0}

    ordered = df.sort_values(["vid", "frame"]).reset_index(drop=True)
    order_votes = votes[df.sort_values(["vid", "frame"]).index.to_numpy()]
    is_blank = ordered.luminance.to_numpy() < 5

    result: dict[str, Any] = {"n_blank": int(blank.sum()), "criteria": {}}
    for j, criterion in enumerate(CRITERIA):
        y = (order_votes[:, :, j].sum(axis=1) >= 2).astype(int)
        rows = []
        for pos in np.flatnonzero(is_blank):
            vid = ordered.vid.iloc[pos]
            same = np.flatnonzero(ordered.vid.to_numpy() == vid)
            local = np.flatnonzero(same == pos)[0]
            # Nearest annotated non-blank neighbour within the same video.
            neighbours = [k for k in same if not is_blank[k]]
            if not neighbours:
                continue
            nearest = min(neighbours, key=lambda k: abs(k - pos))
            rows.append({
                "label": int(y[pos]),
                "neighbour_label": int(y[nearest]),
                "distance": int(abs(nearest - pos)),
                "relative_position": float(local / max(len(same) - 1, 1)),
            })
        if not rows:
            continue
        pos_rows = [r for r in rows if r["label"] == 1]
        neg_rows = [r for r in rows if r["label"] == 0]
        result["criteria"][criterion] = {
            "n": len(rows),
            "positive_blank": len(pos_rows),
            "positive_matching_neighbour": sum(
                1 for r in pos_rows if r["neighbour_label"] == 1),
            "negative_matching_neighbour": sum(
                1 for r in neg_rows if r["neighbour_label"] == 0),
            "mean_relative_position_positive": (
                float(np.mean([r["relative_position"] for r in pos_rows]))
                if pos_rows else float("nan")),
            "mean_relative_position_negative": (
                float(np.mean([r["relative_position"] for r in neg_rows]))
                if neg_rows else float("nan")),
        }
    return result


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metadata", required=True, help="all_metadata.csv")
    p.add_argument("--manifest", default=None,
                   help="endoscapes_frames.csv, for relative_path; enables the "
                        "blank-frame test")
    p.add_argument("--dataset-root", default=None)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    df = pd.read_csv(args.metadata)
    df = df[df.cvs_annotator_1.notna()].reset_index(drop=True)
    votes, df = per_rater_matrix(df)
    print(f"{len(df)} frames with all three raters, {df.vid.nunique()} videos\n")

    results: dict[str, Any] = {}

    print("=" * 72)
    print("1. Is disagreement nested?")
    print("=" * 72)
    results["nesting"] = test_nesting(votes)
    for criterion, entry in results["nesting"].items():
        rates = ", ".join(f"r{r} {entry['rates'][r]:.3f}" for r in RATERS)
        print(f"\n{criterion}   rates: {rates}")
        print(f"       strict to lenient: {entry['strict_to_lenient']}")
        print(f"  {'pair':<8}{'violations':>12}{'expected':>11}{'ratio':>8}"
              f"{'% of strict positives':>24}")
        for pair, v in entry["pairs"].items():
            print(f"  {pair:<8}{v['strict_yes_lenient_no']:>12}"
                  f"{v['expected_if_independent']:>11.1f}{v['ratio']:>8.2f}"
                  f"{100*v['violation_share_of_strict_positives']:>23.1f}%")
    print("\n  A ratio near zero is a threshold. A ratio near one is independence.")
    print("  Values between say the bars overlap but are not the whole story.")

    print("\n" + "=" * 72)
    print("2. Are isolated positives rarer per rater than in the majority?")
    print("=" * 72)
    results["isolation"] = test_isolation(votes, df)
    for criterion, entry in results["isolation"].items():
        print(f"\n{criterion}")
        print(f"  {'source':<12}{'runs':>7}{'isolated':>10}{'share':>9}{'median':>9}")
        for key in (f"rater_{r}" for r in RATERS):
            v = entry[key]
            print(f"  {key:<12}{v['n_runs']:>7}{v['isolated']:>10}"
                  f"{100*v['isolated_share']:>8.1f}%{v['median_run']:>9.1f}")
        v = entry["majority"]
        print(f"  {'majority':<12}{v['n_runs']:>7}{v['isolated']:>10}"
              f"{100*v['isolated_share']:>8.1f}%{v['median_run']:>9.1f}")
    print("\n  The majority being rougher than every individual rater is the")
    print("  episode account: the vote manufactures flicker at episode edges.")
    print("  Independent noise predicts the opposite, since averaging smooths.")

    if args.manifest and args.dataset_root:
        print("\n" + "=" * 72)
        print("3. Do blank frames carry their stretch's label?")
        print("=" * 72)
        manifest = pd.read_csv(args.manifest)
        results["blank"] = test_blank_frames(
            votes, df, Path(args.dataset_root), manifest)
        blank = results["blank"]
        print(f"\n{blank.get('n_blank', 0)} blank frames")
        for criterion, v in blank.get("criteria", {}).items():
            print(f"\n{criterion}  {v['positive_blank']} positive of {v['n']}")
            print(f"  positive blanks whose nearest non-blank neighbour is "
                  f"positive: {v['positive_matching_neighbour']}"
                  f" of {v['positive_blank']}")
            print(f"  negative blanks whose nearest neighbour is negative:  "
                  f"{v['negative_matching_neighbour']}")
            print(f"  mean position in video: positive "
                  f"{v['mean_relative_position_positive']:.2f}, negative "
                  f"{v['mean_relative_position_negative']:.2f}")
        print("\n  Positive blanks beside positive neighbours, and later in the")
        print("  video than negative ones, is inheritance. Uniform position and")
        print("  neighbour labels at the base rate is noise.")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "threshold_account_tests.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwritten to {out / 'threshold_account_tests.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
