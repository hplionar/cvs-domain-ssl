#!/usr/bin/env python3
"""Token structure across three objectives, before and after adaptation.

The quantitative claim is carried by within-frame effective rank and pairwise
cosine, reported in the results chapter. This figure exists because those are
two numbers and the thing they describe is spatial, and a reader who has been
told that DINOv2 lost 30% of its within-frame rank has no picture of what that
means until they see it.

Three arms, one row each, base beside adapted on the same frames:

    DINOv2      loses differentiation; the adapted panels tend toward a single
                colour per frame while remaining distinct between frames, which
                is why a batch-level collapse detector sees nothing wrong
    DINOv3      develops a positional gradient; its second principal component
                correlates 0.918 with patch position after adaptation, so the
                panels shade smoothly top to bottom rather than following
                anatomy
    ViT-MAE     changes very little, and is the only arm of the three to
                improve downstream

**The same frames are used for every arm**, which the earlier per-arm figures
did not do. Comparing three arms on three different random draws confounds the
arm with the frame, and the whole point is a like-for-like comparison.

**Colours are not comparable between panels of different arms.** Each arm's PCA
basis is fitted on its own features, and continued pretraining rotates the
feature space, so a red region in one panel has no relation to a red region in
another. What is comparable is whether coherent regions exist at all, and
whether the same regions persist between an arm's base and adapted panels.
The caption must say this; a reader who assumes otherwise will read the figure
backwards.

Usage:
    python eval/figure_structure.py \\
        --cache-root ../cache --dataset-root ../datasets/SAGES_CVS_Challenge_2024 \\
        --manifest metadata/sages_frames_internal_split.csv \\
        --sample-ids <id1> <id2> \\
        --output-dir ../outputs/figures
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

#: base, adapted, display name. The 560-video adapted arms are used throughout,
#: since they match the corpus every other result in the dissertation uses; the
#: 700-video DINOv2 arm shows a larger change and is mentioned in the caption
#: rather than plotted, because it is transductive.
ARMS = [
    ("dinov2_b", "dinov2_b_trainonly", "DINOv2", "self-distillation"),
    ("dinov3_b", "dinov3_b_trainonly", "DINOv3", "self-distillation"),
    ("mae_b", "mae_b_trainonly", "ViT-MAE", "masked reconstruction"),
]


def read_index(cache: Path) -> list[str]:
    with open(cache / "index.csv", newline="", encoding="utf-8") as fh:
        return [r["sample_id"] for r in csv.DictReader(fh)]


def grid_of(cache: Path) -> tuple[int, int]:
    g = json.loads((cache / "manifest.json").read_text())["encoder"]["token_layout"]["grid"]
    return int(g[0]), int(g[1])


def pca_rgb(tokens: np.ndarray, mean: np.ndarray, comps: np.ndarray,
            lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    proj = (tokens - mean) @ comps.T
    return np.clip((proj - lo) / (hi - lo + 1e-8), 0.0, 1.0)


def fit_pca(flat: np.ndarray, n: int = 3):
    """Principal directions, with signs fixed so repeated runs agree.

    The sign of a principal component is arbitrary, and leaving it so would mean
    the figure changed colour between regenerations for no reason a reader could
    see.
    """
    mean = flat.mean(axis=0, keepdims=True)
    centred = flat - mean
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    comps = vt[:n]
    for i in range(n):
        if (centred @ comps[i]).mean() < 0:
            comps[i] = -comps[i]
    return mean, comps


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache-root", default="../cache")
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--sample-ids", nargs="+", default=None,
                   help="two frames, used identically for every arm; chosen "
                        "rather than drawn so that the comparison is like for "
                        "like and the frames show visible anatomy")
    p.add_argument("--n-frames", type=int, default=2)
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

    cache_root = Path(args.cache_root)
    meta = pd.read_csv(args.manifest).set_index("sample_id")

    # Every arm must hold the same samples in the same order, or the figure
    # compares different frames while appearing to compare arms.
    reference = None
    for base, adapted, _, _ in ARMS:
        for arm in (base, adapted):
            ids = read_index(cache_root / arm / "sages" / args.split)
            if reference is None:
                reference = ids
            elif ids != reference:
                raise SystemExit(f"{arm} holds a different sample order.")
    assert reference is not None

    if args.sample_ids:
        chosen = [s for s in args.sample_ids if s in reference]
        missing = set(args.sample_ids) - set(chosen)
        if missing:
            raise SystemExit(f"not in the cache: {sorted(missing)}")
    else:
        rows = meta.loc[reference].reset_index()
        achieved = rows[[f"c{i}_consensus" for i in (1, 2, 3)]].sum(axis=1)
        pool = rows.index[achieved >= 2].to_numpy()
        rng = np.random.default_rng(args.seed)
        chosen = [reference[i] for i in sorted(rng.choice(pool, args.n_frames, replace=False))]
    print("frames:", chosen)

    order = {s: i for i, s in enumerate(reference)}
    rows_idx = [order[s] for s in chosen]

    n_frames = len(chosen)
    n_cols = 1 + 2 * len(ARMS)
    fig, axes = plt.subplots(n_frames, n_cols,
                             figsize=(args.width, args.width * n_frames / n_cols * 1.06),
                             squeeze=False)

    root = Path(args.dataset_root)
    for r, sample in enumerate(chosen):
        path = root / str(meta.loc[sample, "relative_path"])
        img = (np.asarray(Image.open(path).convert("RGB").resize((224, 224), Image.BICUBIC))
               if path.is_file() else np.zeros((224, 224, 3), np.uint8))
        axes[r][0].imshow(img)
        axes[r][0].set_xticks([]); axes[r][0].set_yticks([])
        if r == 0:
            axes[r][0].set_title("frame", fontsize=8.5)

    for a, (base, adapted, name, objective) in enumerate(ARMS):
        for k, arm in enumerate((base, adapted)):
            col = 1 + 2 * a + k
            cache = cache_root / arm / "sages" / args.split
            grid = grid_of(cache)
            tokens = np.asarray(np.load(cache / "tokens.npy", mmap_mode="r")[rows_idx],
                                dtype=np.float32)
            # One basis per arm across the frames shown, so colours mean the
            # same thing down a column.
            flat = tokens.reshape(-1, tokens.shape[-1])
            mean, comps = fit_pca(flat)
            proj = (flat - mean) @ comps.T
            lo, hi = np.percentile(proj, 2, axis=0), np.percentile(proj, 98, axis=0)

            for r in range(n_frames):
                ax = axes[r][col]
                rgb = pca_rgb(tokens[r], mean, comps, lo, hi)
                ax.imshow(rgb.reshape(grid + (3,)), interpolation="nearest")
                ax.set_xticks([]); ax.set_yticks([])
                if r == 0:
                    ax.set_title("base" if k == 0 else "adapted", fontsize=8)

        # A label spanning the arm's two columns, placed above the panel titles.
        left = axes[0][1 + 2 * a].get_position()
        right = axes[0][2 + 2 * a].get_position()
        fig.text((left.x0 + right.x1) / 2, right.y1 + 0.055, name,
                 ha="center", va="bottom", fontsize=9)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_structure.{ext}", dpi=200)
    plt.close(fig)
    print(f"written to {out/'fig_structure.pdf'}")

    print("\nSuggested caption:")
    print("""
The first three principal components of the patch tokens, rendered as red,
green and blue, for the same two frames under three pretext objectives before
and after continued pretraining on cholecystectomy video. Colours are fitted per
arm and are not comparable between arms, since continued pretraining rotates the
feature space; what is comparable is whether coherent regions persist. DINOv2
loses regional structure while remaining distinct between frames, which is why a
collapse detector operating on pooled outputs across a batch registers nothing.
DINOv3 develops a smooth gradient whose second principal component correlates
0.918 with patch position. ViT-MAE changes least and is the only one of the
three to improve downstream. The DINOv2 arm shown is adapted on the 560-video
corpus; the 700-video arm shows a larger change and is transductive on this
split.
""".strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
