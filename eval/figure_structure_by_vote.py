#!/usr/bin/env python3
"""Token structure grouped by annotator vote count.

Every encoder tested performs 0.15 to 0.32 AUC worse on frames where one of
three annotators dissented, on both benchmarks and without exception. Two
explanations remain open and the released annotations cannot separate them: the
frames may be harder to see, or the label may be closer to arbitrary.

This figure asks a question that sits between them and is answerable. If the
encoder's representation of a contested frame is itself less structured -- fewer
independent directions among its patch tokens, less regional coherence -- then
the difficulty is visible in the representation and not only in the label. If
contested frames are represented just as richly as unanimous ones, then whatever
distinguishes them is not something the encoder fails to encode.

Four rows, one per vote count: none of three annotators judged the criterion
met, one did, two did, all three did. The first and last are the unanimous
stratum and the middle two the contested one. Each row shows frames drawn at
random within its group, with the source beside the first three principal
components of its patch tokens.

The printed summary is the part to report. The panels illustrate; the effective
rank and cosine per vote group are the measurement, and they carry a bootstrap
interval over videos because the four groups differ greatly in size.

Usage:
    python eval/figure_structure_by_vote.py \\
        --arm dinov2_b --criterion c1 \\
        --cache ../cache/dinov2_b/sages/val \\
        --dataset-root ../datasets/SAGES_CVS_Challenge_2024 \\
        --manifest metadata/sages_frames_internal_split.csv \\
        --output-dir ../outputs/figures
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

RATERS = (1, 2, 3)
GROUPS = [(0, "0 of 3", "unanimous"), (1, "1 of 3", "contested"),
          (2, "2 of 3", "contested"), (3, "3 of 3", "unanimous")]


def read_index(cache: Path) -> list[str]:
    with open(cache / "index.csv", newline="", encoding="utf-8") as fh:
        return [r["sample_id"] for r in csv.DictReader(fh)]


def fit_pca(flat: np.ndarray, n: int = 3):
    mean = flat.mean(axis=0, keepdims=True)
    centred = flat - mean
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    comps = vt[:n]
    for i in range(n):
        if (centred @ comps[i]).mean() < 0:
            comps[i] = -comps[i]
    return mean, comps


def frame_statistics(tokens: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean pairwise cosine and effective rank, per frame.

    Effective rank is computed on the centred token covariance: the mean token
    is what pooling discards, and an uncentred spectrum would be dominated by
    it.
    """
    cos = np.empty(len(tokens))
    rank = np.empty(len(tokens))
    for i, h in enumerate(tokens):
        norms = np.linalg.norm(h, axis=1, keepdims=True)
        unit = h / (norms + 1e-8)
        p = h.shape[0]
        cos[i] = (float((unit @ unit.T).sum()) - p) / (p * (p - 1))
        centred = h - h.mean(axis=0, keepdims=True)
        eig = np.clip(np.linalg.eigvalsh(centred @ centred.T), 0.0, None)
        spec = eig / (eig.sum() + 1e-12)
        rank[i] = np.exp(-(spec * np.log(spec + 1e-12)).sum())
    return cos, rank


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--adapted-cache", default=None,
                   help="the same arm after continued pretraining. When given, "
                        "the change in token geometry is reported per vote "
                        "group, which asks whether adaptation costs more "
                        "structure on the frames annotators disagreed about "
                        "than on the frames they agreed about.")
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--criterion", default="c1", choices=("c1", "c2", "c3"))
    p.add_argument("--n-per-group", type=int, default=3)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--width", type=float, default=6.3)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "Liberation Serif",
                       "FreeSerif", "DejaVu Serif"],
        "font.size": 8, "axes.titlesize": 8.5,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })

    cache = Path(args.cache)
    ids = read_index(cache)
    grid = tuple(json.loads((cache / "manifest.json").read_text())
                 ["encoder"]["token_layout"]["grid"])

    meta = pd.read_csv(args.manifest).set_index("sample_id")
    missing = [s for s in ids if s not in meta.index]
    if missing:
        raise SystemExit(f"{len(missing)} sample_ids absent from the manifest.")
    rows = meta.loc[ids].reset_index()

    cols = [f"{args.criterion}_rater{r}" for r in RATERS]
    if all(c in rows.columns for c in cols):
        votes = rows[cols].to_numpy(dtype=int).sum(axis=1)
    else:
        votes = rows[f"{args.criterion}_manual_votes"].to_numpy(dtype=int)

    tokens_all = np.load(cache / "tokens.npy", mmap_mode="r")

    # One PCA basis for the whole arm, so that colours mean the same thing in
    # every panel of the figure. Fitting per group would make the four rows
    # incomparable, which is exactly the comparison the figure is for.
    rng = np.random.default_rng(args.seed)
    fit_rows = rng.choice(len(ids), size=min(200, len(ids)), replace=False)
    flat = np.asarray(tokens_all[sorted(fit_rows)], dtype=np.float32)
    flat = flat.reshape(-1, flat.shape[-1])
    mean, comps = fit_pca(flat)
    proj = (flat - mean) @ comps.T
    lo, hi = np.percentile(proj, 2, axis=0), np.percentile(proj, 98, axis=0)

    # --- the measurement -------------------------------------------------
    print(f"{args.arm}, {args.criterion.upper()}, {len(ids)} frames\n")
    print(f"{'votes':<8}{'n':>6}{'cosine':>10}{'eff. rank':>11}{'% of P':>9}")
    stats: dict[int, dict[str, Any]] = {}
    for value, label, stratum in GROUPS:
        sel = np.flatnonzero(votes == value)
        if sel.size == 0:
            print(f"{label:<8}{0:>6}"); continue
        block = np.asarray(tokens_all[sel], dtype=np.float32)
        cos, rank = frame_statistics(block)
        stats[value] = {"n": int(sel.size), "cosine": float(cos.mean()),
                        "rank": float(rank.mean()),
                        "videos": rows.video_id.to_numpy()[sel],
                        "cos_all": cos, "rank_all": rank}
        print(f"{label:<8}{sel.size:>6}{cos.mean():>10.4f}{rank.mean():>11.2f}"
              f"{100*rank.mean()/grid[0]/grid[1]:>8.1f}%")

    # Contested against unanimous, resampled over videos, because the frames of
    # one procedure are not independent and the four groups differ in size.
    un = np.isin(votes, (0, 3))
    if un.sum() and (~un).sum():
        vids = rows.video_id.to_numpy()
        unique = np.unique(vids)
        by_video = {v: np.flatnonzero(vids == v) for v in unique}
        cos_all = np.concatenate([stats[v]["cos_all"] for v in sorted(stats)])
        order = np.concatenate([np.flatnonzero(votes == v) for v in sorted(stats)])
        cos_by_row = np.empty(len(ids)); cos_by_row[order] = cos_all
        rank_all = np.concatenate([stats[v]["rank_all"] for v in sorted(stats)])
        rank_by_row = np.empty(len(ids)); rank_by_row[order] = rank_all

        boot_c, boot_r = [], []
        rng2 = np.random.default_rng(args.seed + 1)
        for _ in range(args.n_boot):
            pick = np.concatenate([by_video[v] for v in
                                   rng2.choice(unique, unique.size, replace=True)])
            u, c = un[pick], ~un[pick]
            if u.sum() and c.sum():
                boot_c.append(cos_by_row[pick][c].mean() - cos_by_row[pick][u].mean())
                boot_r.append(rank_by_row[pick][c].mean() - rank_by_row[pick][u].mean())
        print(f"\ncontested minus unanimous, {args.n_boot} video-clustered replicates")
        for name, draws, point in (("cosine", boot_c,
                                    cos_by_row[~un].mean() - cos_by_row[un].mean()),
                                   ("eff. rank", boot_r,
                                    rank_by_row[~un].mean() - rank_by_row[un].mean())):
            d = np.array(draws)
            print(f"  {name:<10}{point:>+9.4f}   [{np.percentile(d, 2.5):+.4f}, "
                  f"{np.percentile(d, 97.5):+.4f}]")
        print("\n  An interval excluding zero means contested frames are")
        print("  represented differently, not merely labelled differently, and")
        print("  the difficulty is partly visible in the representation.")
        print("  An interval spanning zero means the encoder represents them")
        print("  as richly as any other frame and the difference is in the label.")

    # --- adaptation, per vote group --------------------------------------
    if args.adapted_cache:
        ad_cache = Path(args.adapted_cache)
        if read_index(ad_cache) != ids:
            raise SystemExit("The adapted cache holds a different sample order.")
        ad_grid = tuple(json.loads((ad_cache / "manifest.json").read_text())
                        ["encoder"]["token_layout"]["grid"])
        if ad_grid != grid:
            raise SystemExit(
                f"Token grids differ, {grid} against {ad_grid}. Effective rank "
                f"is bounded by the number of patches, so the two are not "
                f"comparable."
            )
        ad_tokens = np.load(ad_cache / "tokens.npy", mmap_mode="r")

        print(f"\nchange under adaptation, by vote group")
        print(f"{'votes':<8}{'n':>6}{'cosine':>20}{'eff. rank':>20}")
        changes: dict[int, dict[str, float]] = {}
        for value, label, stratum in GROUPS:
            sel = np.flatnonzero(votes == value)
            if sel.size == 0:
                continue
            base_cos, base_rank = stats[value]["cos_all"], stats[value]["rank_all"]
            ad_cos, ad_rank = frame_statistics(
                np.asarray(ad_tokens[sel], dtype=np.float32))
            dc = float(ad_cos.mean() - base_cos.mean())
            dr = float(ad_rank.mean() - base_rank.mean())
            changes[value] = {"cosine": dc, "rank": dr}
            print(f"{label:<8}{sel.size:>6}"
                  f"{base_cos.mean():>9.4f} -> {ad_cos.mean():.4f}"
                  f"{base_rank.mean():>11.2f} -> {ad_rank.mean():.2f}")

        # The comparison that matters is between strata, not within one. If
        # adaptation costs the same structure everywhere, the geometry change
        # cannot explain why the stratified performance gap persists.
        if 0 in changes and 2 in changes:
            # Weighted by frame count. The groups differ by a factor of
            # twenty in size, and an unweighted mean lets the 54-frame
            # unanimous-positive group count as much as the 1,001-frame
            # unanimous-negative one, which manufactures a difference between
            # strata where none exists.
            def weighted(vs):
                num = sum(changes[v]["rank"] * stats[v]["n"] for v in vs if v in changes)
                den = sum(stats[v]["n"] for v in vs if v in changes)
                return num / den if den else float("nan")
            print(f"\n  mean rank change, weighted by frame count:")
            print(f"    unanimous {weighted((0, 3)):+.2f}  "
                  f"(n = {sum(stats[v]['n'] for v in (0, 3) if v in stats)})")
            print(f"    contested {weighted((1, 2)):+.2f}  "
                  f"(n = {sum(stats[v]['n'] for v in (1, 2) if v in stats)})")
            print("  A larger loss on one stratum would connect the geometry")
            print("  change to the stratified performance gap. A uniform loss")
            print("  means the two findings are independent, and the geometry")
            print("  cannot explain why adaptation fails to help contested frames.")

    # --- the figure ------------------------------------------------------
    n = args.n_per_group
    show_adapted = bool(args.adapted_cache)
    per_sample = 3 if show_adapted else 2

    # The adapted arm needs its own principal basis: continued pretraining
    # rotates the feature space, so projecting its tokens onto the base arm's
    # components would show the rotation rather than the structure. Colours are
    # therefore comparable down a column and between rows, and not between the
    # base and adapted panels of the same frame. What is comparable there is
    # whether coherent regions survive.
    if show_adapted:
        ad_flat = np.asarray(ad_tokens[sorted(fit_rows)], dtype=np.float32)
        ad_flat = ad_flat.reshape(-1, ad_flat.shape[-1])
        ad_mean, ad_comps = fit_pca(ad_flat)
        ad_proj = (ad_flat - ad_mean) @ ad_comps.T
        ad_lo = np.percentile(ad_proj, 2, axis=0)
        ad_hi = np.percentile(ad_proj, 98, axis=0)

    fig, axes = plt.subplots(len(GROUPS), per_sample * n,
                             figsize=(args.width,
                                      args.width * len(GROUPS) / (per_sample * n) * 1.12),
                             squeeze=False)
    root = Path(args.dataset_root)
    for r, (value, label, stratum) in enumerate(GROUPS):
        sel = np.flatnonzero(votes == value)
        pick = (rng.choice(sel, size=min(n, sel.size), replace=False)
                if sel.size else np.array([], dtype=int))
        for c in range(n):
            cells = [axes[r][per_sample * c + k] for k in range(per_sample)]
            for ax in cells:
                ax.set_xticks([]); ax.set_yticks([])
            if c >= pick.size:
                for ax in cells:
                    ax.axis("off")
                continue
            i = int(pick[c])
            path = root / str(rows.loc[i, "relative_path"])
            img = (np.asarray(Image.open(path).convert("RGB").resize((224, 224), Image.BICUBIC))
                   if path.is_file() else np.zeros((224, 224, 3), np.uint8))
            cells[0].imshow(img)
            tok = np.asarray(tokens_all[i], dtype=np.float32)
            cells[1].imshow(
                np.clip(((tok - mean) @ comps.T - lo) / (hi - lo + 1e-8), 0, 1)
                .reshape(grid + (3,)), interpolation="nearest")
            if show_adapted:
                ad_tok = np.asarray(ad_tokens[i], dtype=np.float32)
                cells[2].imshow(
                    np.clip(((ad_tok - ad_mean) @ ad_comps.T - ad_lo)
                            / (ad_hi - ad_lo + 1e-8), 0, 1)
                    .reshape(grid + (3,)), interpolation="nearest")
            if r == 0 and c == 0:
                cells[0].set_title("frame", fontsize=7.5)
                cells[1].set_title("base", fontsize=7.5)
                if show_adapted:
                    cells[2].set_title("adapted", fontsize=7.5)
        axes[r][0].set_ylabel(f"{label}\n{stratum}", fontsize=8)

    title = f"{args.criterion.upper()} by annotator vote count, {args.arm.replace('_', ' ')}"
    if show_adapted:
        title += ", before and after continued pretraining"
    fig.suptitle(title, fontsize=9.5)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    stem = (f"fig_structure_votes_{args.arm}_{args.criterion}"
            + ("_adapt" if show_adapted else ""))
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{stem}.{ext}", dpi=200)
    plt.close(fig)
    print(f"\nwritten to {out / (stem + '.pdf')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
