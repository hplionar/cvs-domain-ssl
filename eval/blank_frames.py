#!/usr/bin/env python3
"""Frames carrying no image content, and what the benchmark does with them.

A random draw of Endoscapes test frames turned up one that was entirely black.
Measuring mean luminance across the split found 136 of 1,799 -- 7.6% -- below a
luminance of 5 out of 255, with identical counts at thresholds of 5, 10 and 20.
The sharpness of that cliff matters: these are not dim frames, they are blank.

Every published result on this benchmark is computed over them. If they are
labelled negative on all three criteria, they are free specificity for every
method, inflating AUC and lowering the chance line for average precision without
any method having done anything. If any are labelled positive, that is a
labelling error in a benchmark the field treats as reference.

The script reports the count, how they distribute across videos, their labels
and their vote patterns, and writes a contact sheet so the frames can be
inspected rather than trusted to a threshold. It also recomputes each arm's
metrics with them removed, which is the number that says whether they matter.

Luminance is computed on a 64x64 downsample of the greyscale frame, which is
enough to separate blank from dark and fast enough for a whole split.

Usage:
    python eval/blank_frames.py \\
        --manifest metadata/endoscapes_frames.csv \\
        --dataset-root ../datasets/endoscapes \\
        --split test --output-dir ../outputs/blank_frames \\
        --probe-dir ../outputs/cvs-domain-ssl/probe/dinov2_b_endoscapes_mean \\
        --cache ../cache/dinov2_b/endoscapes/test
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


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def read_index(cache_dir: Path) -> list[str]:
    with open(cache_dir / "index.csv", newline="", encoding="utf-8") as fh:
        return [r["sample_id"] for r in csv.DictReader(fh)]


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


def luminance(root: Path, paths: list[str]) -> np.ndarray:
    out = np.full(len(paths), np.nan)
    for i, rp in enumerate(paths):
        p = root / rp
        if p.is_file():
            out[i] = np.asarray(
                Image.open(p).convert("L").resize((64, 64))
            ).mean()
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True)
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--split-column", default=None)
    p.add_argument("--threshold", type=float, default=5.0,
                   help="mean luminance below which a frame is treated as blank")
    p.add_argument("--n-show", type=int, default=12)
    p.add_argument("--probe-dir", default=None)
    p.add_argument("--cache", default=None)
    p.add_argument("--logits-prefix", default="test_logits")
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    df = pd.read_csv(args.manifest)
    if "is_cvs_annotated" in df.columns:
        df = df[df.is_cvs_annotated]
    column = args.split_column or ("internal_split" if "internal_split" in df.columns else "split")
    if args.split != "all":
        df = df[df[column] == args.split]
    df = df.reset_index(drop=True)

    root = Path(args.dataset_root)
    lum = luminance(root, [str(x) for x in df.relative_path])
    blank = lum < args.threshold

    print(f"{Path(args.manifest).name}, {column}={args.split}: {len(df)} frames, "
          f"{df.video_id.nunique()} videos")
    print(f"mean luminance {np.nanmean(lum):.1f} of 255\n")

    print(f"{'threshold':>10}{'frames':>9}{'share':>9}")
    for t in (2, 5, 10, 20, 30, 50):
        n = int((lum < t).sum())
        print(f"{t:>10}{n:>9}{100*n/len(lum):>8.1f}%")
    print(f"\n  A count that does not change between thresholds means the frames are")
    print(f"  blank rather than dark, and that a threshold anywhere in that range")
    print(f"  selects the same set.")

    results: dict[str, Any] = {
        "split": args.split, "n_frames": int(len(df)),
        "threshold": args.threshold,
        "n_blank": int(blank.sum()),
        "share_blank": float(blank.mean()),
        "n_videos_affected": int(df[blank].video_id.nunique()),
        "n_videos_total": int(df.video_id.nunique()),
    }

    print(f"\nblank frames (luminance < {args.threshold:g}): {int(blank.sum())} "
          f"of {len(df)} ({100*blank.mean():.1f}%), "
          f"across {df[blank].video_id.nunique()} of {df.video_id.nunique()} videos")

    per_video = df[blank].video_id.value_counts()
    if len(per_video):
        print(f"  worst affected: " + ", ".join(
            f"{v} ({n})" for v, n in per_video.head(5).items()))
        results["per_video"] = {str(k): int(v) for k, v in per_video.items()}

    print(f"\nlabels on blank frames")
    print(f"{'crit':<5}{'positive':>10}{'share':>9}{'overall':>10}{'votes':>22}")
    results["labels"] = {}
    for c in CRITERIA:
        votes = votes_for(df, c)
        y = (votes >= 2).astype(int)
        dist = pd.Series(votes[blank]).value_counts().sort_index().to_dict()
        results["labels"][c] = {
            "positive_blank": int(y[blank].sum()),
            "share_blank": float(y[blank].mean()) if blank.sum() else float("nan"),
            "share_overall": float(y.mean()),
            "vote_distribution": {int(k): int(v) for k, v in dist.items()},
        }
        print(f"{c:<5}{int(y[blank].sum()):>10}{100*y[blank].mean():>8.1f}%"
              f"{100*y.mean():>9.1f}%   {dist}")

    print(f"\n  A blank frame labelled positive is a labelling error: no criterion")
    print(f"  can be satisfied in an image with no content. A blank frame labelled")
    print(f"  negative is free specificity for every method scored on this split.")

    # What the frames cost, if an arm is supplied.
    if args.probe_dir and args.cache:
        from sklearn.metrics import average_precision_score, roc_auc_score

        ids = read_index(Path(args.cache))
        scores = load_scores(Path(args.probe_dir), args.logits_prefix)
        if len(ids) != len(df):
            order = {s: i for i, s in enumerate(ids)}
            keep = [order[s] for s in df.sample_id if s in order]
            if len(keep) != len(df):
                raise SystemExit("The cache and the manifest selection do not align.")
            scores = scores[keep]

        print(f"\nmetrics with and without the blank frames")
        print(f"{'crit':<5}{'AP all':>9}{'AP kept':>10}{'change':>9}"
              f"{'AUC all':>10}{'AUC kept':>10}{'change':>9}")
        results["impact"] = {}
        for j, c in enumerate(CRITERIA):
            y = (votes_for(df, c) >= 2).astype(int)
            s = scores[:, j]
            ap_all = average_precision_score(y, s)
            auc_all = roc_auc_score(y, s)
            ap_kept = average_precision_score(y[~blank], s[~blank])
            auc_kept = roc_auc_score(y[~blank], s[~blank])
            results["impact"][c] = {"ap_all": ap_all, "ap_kept": ap_kept,
                                    "auc_all": auc_all, "auc_kept": auc_kept}
            print(f"{c:<5}{ap_all:>9.4f}{ap_kept:>10.4f}{ap_kept-ap_all:>+9.4f}"
                  f"{auc_all:>10.4f}{auc_kept:>10.4f}{auc_kept-auc_all:>+9.4f}")

        print(f"\n  Removing the blank frames raises the chance line, because it removes")
        print(f"  easy negatives, so AP falling is expected and is the size of the")
        print(f"  free credit. AUC falling by more than a little would mean the frames")
        print(f"  were carrying a meaningful share of the ranking.")

        print(f"\n  mean model probability on blank frames")
        for j, c in enumerate(CRITERIA):
            print(f"    {c}  {scores[blank, j].mean():.4f}   "
                  f"(all frames {scores[:, j].mean():.4f})")

    # Contact sheet.
    idx = np.flatnonzero(blank)[: args.n_show]
    if idx.size:
        cols = 6
        rows = int(np.ceil(idx.size / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 2.4 * rows),
                                 squeeze=False)
        for k in range(rows * cols):
            ax = axes[k // cols][k % cols]
            ax.set_xticks([]); ax.set_yticks([])
            if k >= idx.size:
                ax.axis("off"); continue
            i = idx[k]
            path = root / str(df.loc[i, "relative_path"])
            img = (np.asarray(Image.open(path).convert("RGB")) if path.is_file()
                   else np.zeros((64, 64, 3), dtype=np.uint8))
            ax.imshow(img)
            votes = [int(votes_for(df, c)[i]) for c in CRITERIA]
            ax.set_xlabel(f"lum {lum[i]:.1f}   votes {votes}", fontsize=7)
            ax.set_title(str(df.loc[i, "sample_id"]), fontsize=7)
        fig.suptitle(
            f"Frames with mean luminance below {args.threshold:g} of 255 "
            f"({int(blank.sum())} of {len(df)}, {100*blank.mean():.1f}%)",
            fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"blank_frames_{args.split}.png"
        fig.savefig(path, dpi=140, bbox_inches="tight")
        print(f"\ncontact sheet written to {path}")

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"blank_frames_{args.split}.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    # The list itself, so that any arm can be rescored without them.
    listing = out_dir / f"blank_sample_ids_{args.split}.txt"
    listing.write_text("\n".join(df.sample_id[blank].astype(str)) + "\n")
    print(f"sample ids written to {listing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
