#!/usr/bin/env python3
"""What does a contested frame actually look like?

Every quantitative route to this question is closed. A ceiling restricted to
contested frames is circular, since contested means a 2-1 split and any measure
of agreement on them is fixed by that definition rather than estimated. The
disagreement probe found that apparent predictability of disagreement was
criterion strength in disguise. So whether the contested residual reflects
irreducible ambiguity or a gap no method has closed cannot be settled by
measurement from the released annotations.

It can be looked at. This script lays out frames by vote count, so that the four
states -- all three annotators negative, one dissenting, two agreeing positive,
all three positive -- can be compared side by side with the model's score for
each.

Two things to look for, and they point in opposite directions:

    contested frames are visibly obscured    smoke, blood, instruments across
                                             the field, partial exposure. The
                                             frames are hard, the annotators
                                             disagree because the evidence is
                                             genuinely weak, and better inputs
                                             might help.
    contested frames look clear              the anatomy is visible and the
                                             surgeons still disagreed, so the
                                             disagreement is about judgement
                                             rather than visibility, and no
                                             change to the input will resolve it.

This is an observation, not a measurement, and should be written as one. Its
value is that it constrains which explanation is plausible, and a reader can
check it against the figure.

Frames are drawn at random within each vote group rather than selected by model
score, so the figure does not become a display of the model's best and worst
cases. Pass --sort-by-score to see those instead, which is a different question:
where is the model confidently wrong.

Usage:
    python eval/inspect_strata.py \\
        --manifest metadata/sages_frames_official_test.csv \\
        --dataset-root ../datasets/SAGES_CVS_Challenge_2024 \\
        --criterion c1 --n-per-group 6 \\
        --probe-dir ../outputs/cvs-domain-ssl/probe/dinov2_b_sages_mean \\
        --cache ../cache/dinov2_b/sages_official/test \\
        --logits-prefix test_logits_official \\
        --output-dir ../outputs/strata_inspection
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

CRITERIA = ("c1", "c2", "c3")
RATERS = (1, 2, 3)

GROUPS = [
    (0, "0 of 3", "unanimous negative"),
    (1, "1 of 3", "contested, majority negative"),
    (2, "2 of 3", "contested, majority positive"),
    (3, "3 of 3", "unanimous positive"),
]


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def read_index(cache_dir: Path) -> list[str]:
    with open(cache_dir / "index.csv", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return [r["sample_id"] for r in rows]


def load_scores(probe_dir: Path, prefix: str) -> np.ndarray:
    files = sorted(probe_dir.glob(f"{prefix}_seed*.npz"))
    if not files:
        raise FileNotFoundError(f"No {prefix}_seed*.npz in {probe_dir}")
    return np.mean([sigmoid(np.load(f)["logits"]) for f in files], axis=0)


def votes_for(df: pd.DataFrame, criterion: str) -> np.ndarray:
    cols = [f"{criterion}_rater{r}" for r in RATERS]
    if all(c in df.columns for c in cols):
        return df[cols].to_numpy(dtype=int).sum(axis=1)
    col = f"{criterion}_manual_votes"
    if col in df.columns:
        return df[col].to_numpy(dtype=int)
    raise SystemExit(f"Manifest has neither {cols} nor {col}.")


def load_frame(root: Path, relative_path: str, size: int = 224) -> np.ndarray:
    path = root / relative_path
    if not path.is_file():
        return np.zeros((size, size, 3), dtype=np.uint8)
    return np.asarray(Image.open(path).convert("RGB").resize((size, size), Image.BICUBIC))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True)
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--criterion", default="c1", choices=CRITERIA)
    p.add_argument("--split", default=None,
                   help="filter on the manifest's split column, if it has one")
    p.add_argument("--n-per-group", type=int, default=6)
    p.add_argument("--probe-dir", default=None,
                   help="optional: annotate each frame with the model's score")
    p.add_argument("--cache", default=None,
                   help="the cache the probe was scored on, for sample order")
    p.add_argument("--logits-prefix", default="test_logits")
    p.add_argument("--sort-by-score", action="store_true",
                   help="show the highest-scoring frames of each group rather "
                        "than a random draw. A different question: where is the "
                        "model confidently wrong.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    df = pd.read_csv(args.manifest)
    if "is_cvs_annotated" in df.columns:
        df = df[df.is_cvs_annotated]
    if args.split:
        column = "internal_split" if "internal_split" in df.columns else "split"
        df = df[df[column] == args.split]
    if "relative_path" not in df.columns:
        raise SystemExit("The manifest has no relative_path column.")

    scores = None
    if args.probe_dir and args.cache:
        ids = read_index(Path(args.cache))
        raw = load_scores(Path(args.probe_dir), args.logits_prefix)
        if raw.shape[0] != len(ids):
            raise SystemExit(f"{raw.shape[0]} rows of logits against {len(ids)} "
                             f"cached samples.")
        lookup = {s: raw[i, CRITERIA.index(args.criterion)] for i, s in enumerate(ids)}
        df = df[df.sample_id.isin(lookup)]
        scores = df.sample_id.map(lookup).to_numpy()

    df = df.reset_index(drop=True)
    votes = votes_for(df, args.criterion)
    rng = np.random.default_rng(args.seed)
    root = Path(args.dataset_root)

    n = args.n_per_group
    fig, axes = plt.subplots(4, n, figsize=(2.2 * n, 9.6), squeeze=False)
    summary: dict[str, Any] = {"criterion": args.criterion, "groups": {}}

    for r, (value, label, description) in enumerate(GROUPS):
        pool = np.flatnonzero(votes == value)
        summary["groups"][label] = {"n_available": int(pool.size),
                                    "description": description}
        if pool.size == 0:
            for c in range(n):
                axes[r][c].axis("off")
            axes[r][0].set_ylabel(f"{label}\n(none)", fontsize=9)
            continue

        if args.sort_by_score and scores is not None:
            pick = pool[np.argsort(-scores[pool])][:n]
        else:
            pick = rng.choice(pool, size=min(n, pool.size), replace=False)

        for c in range(n):
            ax = axes[r][c]
            ax.set_xticks([]); ax.set_yticks([])
            if c >= pick.size:
                ax.axis("off"); continue
            i = pick[c]
            ax.imshow(load_frame(root, str(df.loc[i, "relative_path"])))
            if scores is not None:
                ax.set_xlabel(f"{scores[i]:.2f}", fontsize=8)
            if c == 0:
                ax.set_ylabel(f"{label}\n{description}", fontsize=8)

    title = (f"{args.criterion.upper()} by annotator vote count"
             + (f", {args.split}" if args.split else ""))
    if scores is not None:
        title += "   (numbers are the model's probability)"
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"strata_{args.criterion}" + ("_by_score" if args.sort_by_score else "")
    path = out_dir / f"{stem}.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    print(f"written to {path}")

    print(f"\nframes available per group, {args.criterion.upper()}")
    for value, label, description in GROUPS:
        n_avail = int((votes == value).sum())
        print(f"  {label:<8} {n_avail:>6}   {description}")
    contested = int(((votes == 1) | (votes == 2)).sum())
    positives = int((votes >= 2).sum())
    pos_contested = int((votes == 2).sum())
    print(f"\n  contested {contested} of {len(votes)} frames "
          f"({100*contested/len(votes):.1f}%)")
    print(f"  {100*pos_contested/max(positives,1):.1f}% of positive labels are contested")

    if scores is not None:
        print(f"\nmean model probability by vote count")
        for value, label, _ in GROUPS:
            m = votes == value
            if m.sum():
                print(f"  {label:<8} {scores[m].mean():.4f}  (n={int(m.sum())})")
        print("  A monotone rise across the four groups means the model recovers")
        print("  the vote count the binary label discards, which the dissent")
        print("  alignment analysis found on the validation split.")

    with open(out_dir / f"{stem}.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
