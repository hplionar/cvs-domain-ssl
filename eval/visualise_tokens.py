#!/usr/bin/env python3
"""Where in the frame the token geometry changed.

``eval/patch_similarity.py`` reports one number per frame: the mean pairwise
cosine between its patch tokens, and the effective rank of their covariance.
For dinov2_b those moved sharply under continued pretraining -- cosine 0.398 to
0.534, effective rank 30.9 to 19.6, in 98% and 99% of frames respectively. What
a single number cannot say is *where*.

That distinction matters for this task. If differentiation is lost uniformly,
the encoder has become globally less spatially informative. If it is lost in the
background while the hepatocystic region stays differentiated, the same summary
statistic would be produced by a change that leaves the criteria largely
intact. The two have different implications and the aggregate cannot separate
them.

Three panels per frame, base beside adapted:

    source          the frame as the encoder saw it, resized to the model input
    differentiation each patch coloured by its mean cosine to the other patches
                    of the same frame. Low values mean the patch is distinctive;
                    high values mean it resembles everything else. This is the
                    per-patch decomposition of the frame-level number.
    structure       the first three principal components of the patch tokens
                    mapped to red, green and blue. Patches sharing a colour hold
                    similar representations, so anatomy that the encoder
                    distinguishes appears as distinct colour regions.

Two things to keep in mind when reading the structure panels. The PCA basis is
fitted per arm across the sampled frames, so colours are consistent between
frames within an arm and *not* comparable between arms: continued pretraining
rotates the feature space, and a red region in one panel has no relation to a
red region in the other. What is comparable is whether coherent regions exist at
all. Component signs are also arbitrary, and are fixed here by a convention that
makes each component's mean positive, so that repeated runs produce the same
image.

Usage:
    python eval/visualise_tokens.py \\
        --base    ../cache/dinov2_b/sages/val \\
        --adapted ../cache/dinov2_b_adapted/sages/val \\
        --manifest metadata/sages_frames_internal_split.csv \\
        --dataset-root ../datasets/SAGES_CVS_Challenge_2024 \\
        --n-frames 6 --output-dir ../outputs/token_visualisation
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


def read_index(cache_dir: Path) -> list[str]:
    with open(cache_dir / "index.csv", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return [r["sample_id"] for r in rows]


def grid_of(cache_dir: Path) -> tuple[int, int]:
    m = json.loads((cache_dir / "manifest.json").read_text())
    g = m["encoder"]["token_layout"]["grid"]
    if len(g) != 2:
        raise SystemExit(
            f"{cache_dir} has a {len(g)}-dimensional token grid {g}. This "
            f"visualisation is for image encoders; a video cache has no single "
            f"spatial grid per sample."
        )
    return int(g[0]), int(g[1])


def image_size_of(cache_dir: Path) -> int:
    m = json.loads((cache_dir / "manifest.json").read_text())
    return int(m.get("transform", {}).get("image_size", 224))


def differentiation(tokens: np.ndarray) -> np.ndarray:
    """[P] mean cosine of each patch to the others in its frame.

    The diagonal is excluded: a patch's similarity to itself is one by
    construction and would compress the range by 1/P.
    """
    unit = tokens / (np.linalg.norm(tokens, axis=1, keepdims=True) + 1e-8)
    gram = unit @ unit.T
    p = gram.shape[0]
    return (gram.sum(axis=1) - 1.0) / (p - 1)


def fit_pca(stacked: np.ndarray, n_components: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Principal directions of the pooled patch tokens of several frames.

    Fitted once per arm rather than per frame so that colours mean the same
    thing across the frames shown. Component signs are fixed by making each
    component's mean loading positive, since the sign is otherwise arbitrary and
    would flip between runs.
    """
    mean = stacked.mean(axis=0, keepdims=True)
    centred = stacked - mean
    # Economy SVD: the number of tokens far exceeds the components needed.
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    comps = vt[:n_components]
    for i in range(n_components):
        if (centred @ comps[i]).mean() < 0:
            comps[i] = -comps[i]
    return mean, comps


def project_rgb(tokens: np.ndarray, mean: np.ndarray, comps: np.ndarray,
                lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """[P, 3] in [0, 1], using percentile bounds shared across frames."""
    proj = (tokens - mean) @ comps.T
    scaled = (proj - lo) / (hi - lo + 1e-8)
    return np.clip(scaled, 0.0, 1.0)


def load_frame(root: Path, relative_path: str, size: int) -> np.ndarray:
    p = root / relative_path
    if not p.is_file():
        return np.zeros((size, size, 3), dtype=np.uint8)
    img = Image.open(p).convert("RGB").resize((size, size), Image.BICUBIC)
    return np.asarray(img)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", required=True)
    p.add_argument("--adapted", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--n-frames", type=int, default=6)
    p.add_argument("--sample-ids", nargs="*", default=None,
                   help="explicit frames to show; overrides --n-frames")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--label", nargs=2, default=("base", "adapted"),
                   help="panel titles for the two arms")
    args = p.parse_args()

    base_dir, adapt_dir = Path(args.base), Path(args.adapted)
    ids_b, ids_a = read_index(base_dir), read_index(adapt_dir)
    if ids_b != ids_a:
        raise SystemExit("The two caches hold different samples in a different order.")
    grid = grid_of(base_dir)
    if grid_of(adapt_dir) != grid:
        raise SystemExit(
            f"Token grids differ: {grid} against {grid_of(adapt_dir)}. The two "
            f"arms are not spatially comparable."
        )
    size = image_size_of(base_dir)

    meta = pd.read_csv(args.manifest).set_index("sample_id")
    if "relative_path" not in meta.columns:
        raise SystemExit("The manifest has no relative_path column; frames cannot be located.")

    if args.sample_ids:
        chosen = [i for i in args.sample_ids if i in ids_b]
        missing = set(args.sample_ids) - set(chosen)
        if missing:
            print(f"not in this cache, skipped: {sorted(missing)[:5]}")
    else:
        # Prefer frames on which the annotators agreed the criteria were met:
        # the anatomy is then actually visible, which is what the panels are for.
        rows = meta.loc[ids_b].reset_index()
        achieved = rows[[f"{c}_consensus" for c in CRITERIA]].sum(axis=1)
        rng = np.random.default_rng(args.seed)
        pool = rows.index[achieved >= 2].to_numpy()
        if pool.size < args.n_frames:
            pool = np.arange(len(rows))
        pick = rng.choice(pool, size=min(args.n_frames, pool.size), replace=False)
        chosen = [ids_b[i] for i in sorted(pick)]

    order = {s: i for i, s in enumerate(ids_b)}
    rows_idx = [order[s] for s in chosen]
    print(f"grid {grid[0]}x{grid[1]}  input {size}px  {len(chosen)} frames")

    tokens = {}
    for name, d in ((args.label[0], base_dir), (args.label[1], adapt_dir)):
        arr = np.load(d / "tokens.npy", mmap_mode="r")
        tokens[name] = np.asarray(arr[rows_idx], dtype=np.float32)

    # One PCA basis per arm, and shared colour bounds, so that the panels of an
    # arm can be compared with one another.
    pca, bounds = {}, {}
    for name, t in tokens.items():
        flat = t.reshape(-1, t.shape[-1])
        mean, comps = fit_pca(flat)
        proj = (flat - mean) @ comps.T
        bounds[name] = (np.percentile(proj, 2, axis=0), np.percentile(proj, 98, axis=0))
        pca[name] = (mean, comps)

    # Shared colour scale for the differentiation panels: the comparison between
    # arms is the point, so a per-panel scale would hide it.
    diffs = {name: np.stack([differentiation(t[i]) for i in range(len(chosen))])
             for name, t in tokens.items()}
    vmin = min(d.min() for d in diffs.values())
    vmax = max(d.max() for d in diffs.values())

    root = Path(args.dataset_root)
    n = len(chosen)
    fig, axes = plt.subplots(n, 5, figsize=(15, 3.0 * n), squeeze=False)
    titles = ["frame", f"differentiation\n{args.label[0]}",
              f"differentiation\n{args.label[1]}",
              f"structure\n{args.label[0]}", f"structure\n{args.label[1]}"]

    for r, sample in enumerate(chosen):
        row = meta.loc[sample]
        img = load_frame(root, str(row["relative_path"]), size)
        axes[r][0].imshow(img)
        votes = [int(row[f"{c}_consensus"]) for c in CRITERIA]
        axes[r][0].set_ylabel(f"C1{votes[0]} C2{votes[1]} C3{votes[2]}", fontsize=8)

        for col, name in enumerate(args.label, start=1):
            im = axes[r][col].imshow(diffs[name][r].reshape(grid),
                                     cmap="viridis", vmin=vmin, vmax=vmax)
            axes[r][col].set_xlabel(f"mean {diffs[name][r].mean():.3f}", fontsize=8)

        for col, name in enumerate(args.label, start=3):
            mean, comps = pca[name]
            lo, hi = bounds[name]
            rgb = project_rgb(tokens[name][r], mean, comps, lo, hi)
            axes[r][col].imshow(rgb.reshape(grid + (3,)))

        for c in range(5):
            axes[r][c].set_xticks([]); axes[r][c].set_yticks([])
            if r == 0:
                axes[r][c].set_title(titles[c], fontsize=10)

    fig.colorbar(im, ax=axes[:, 1:3].ravel().tolist(), shrink=0.6,
                 label="mean cosine to other patches")
    fig.suptitle(
        f"Within-frame token differentiation and structure, "
        f"{args.label[0]} against {args.label[1]}", fontsize=12)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"tokens_{args.label[0]}_vs_{args.label[1]}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"written to {path}")

    summary: dict[str, Any] = {
        "grid": list(grid), "image_size": size, "sample_ids": chosen,
        "differentiation": {name: {"mean": float(d.mean()),
                                   "min": float(d.min()), "max": float(d.max())}
                            for name, d in diffs.items()},
    }
    # Whether the change is uniform or concentrated: the correlation between the
    # two arms' per-patch differentiation maps. Near 1 means every patch moved
    # by a similar amount; low means the change is localised.
    a, b = args.label
    per_frame = [float(np.corrcoef(diffs[a][i], diffs[b][i])[0, 1])
                 for i in range(len(chosen))]
    summary["spatial_correlation"] = {"per_frame": per_frame,
                                      "mean": float(np.mean(per_frame))}
    print(f"\nspatial correlation of the two differentiation maps: "
          f"{np.mean(per_frame):.3f}")
    print("  Near 1 means every patch changed by a similar amount, so the loss of")
    print("  differentiation is uniform. A low value means it is localised, and")
    print("  the panels show where.")

    with open(out_dir / "token_visualisation.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
