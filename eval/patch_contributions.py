#!/usr/bin/env python3
"""Exact per-patch contributions to each criterion logit, and how adaptation
moved them.

For the mean head the logit is a linear function of the pooled token grid:

    z_c = w_c . (1/N) sum_i h_i + b_c  =  (1/N) sum_i (w_c . h_i) + b_c

so each patch contributes exactly (1/N)(w_c . h_i). This is an identity, not an
attribution approximation: no gradients, no perturbation, no assumption about
local linearity. The contributions sum to the logit minus the bias, which the
script asserts.

What that buys. When continued pretraining changes a criterion's performance,
two very different things could be responsible: the encoder now represents the
relevant structures differently, or the head has reweighted an unchanged
representation. Comparing the *spatial distribution* of contributions between a
base and an adapted arm separates them without masks, without a segmentation
model, and without any GPU:

* If the contribution maps are near-identical, the head is reading the same
  places and any performance change came from how it weights them.
* If the maps move, the encoder's spatial content changed and the head followed.

Three quantities are reported per criterion:

    entropy      of the normalised positive contribution mass over patches.
                 Low entropy means the decision rests on a few patches; high
                 entropy means it is spread across the frame.
    top-k mass   fraction of positive contribution held by the highest k
                 patches, as a concentration measure that does not assume a
                 distribution.
    map shift    mean cosine similarity between the base and adapted
                 contribution maps of the same frame. This is the direct
                 measure of whether the head is reading the same places.

Limits worth stating. This describes the *head's* view, not the encoder's
content: a patch with a large contribution is one the head weights heavily,
which is not the same as one the encoder represents richly. And it applies only
to heads that are linear over pooled tokens -- the mean head. The attentive and
fusion heads admit a similar decomposition with an extra weight per patch, but
that is a different quantity and is not computed here.

Usage:
    python eval/patch_contributions.py \\
        --base    ../outputs/cvs-domain-ssl/probe/dinov2_b_sages_mean:../cache/dinov2_b/sages/val \\
        --adapted ../outputs/cvs-domain-ssl/probe/dinov2_b_adapted_sages_mean:../cache/dinov2_b_adapted/sages/val \\
        --manifest metadata/sages_frames_internal_split.csv \\
        --output-dir ../outputs/patch_contributions
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

CRITERIA = ("c1", "c2", "c3")
RATERS = (1, 2, 3)

#: Bytes of one decoded block while computing contributions.
BLOCK_BYTES = 1 << 30


def read_index(cache_dir: Path) -> list[str]:
    with open(cache_dir / "index.csv", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if [int(r["row"]) for r in rows] != list(range(len(rows))):
        raise ValueError(f"{cache_dir/'index.csv'} rows are not 0..N-1 in order.")
    return [r["sample_id"] for r in rows]


def load_weights(probe_dir: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """The linear map of the mean head, averaged over seeds.

    The head is LayerNorm followed by Linear. LayerNorm is affine per feature,
    so the composition is still linear in the pooled vector, but the
    normalisation statistics are computed from the pooled vector itself and
    therefore differ per sample. The weights are extracted here and the
    normalisation applied per sample in ``contributions``.
    """
    paths = sorted(probe_dir.glob("head_seed*.pt"))
    if not paths:
        raise FileNotFoundError(
            f"No head_seed*.pt in {probe_dir}. This probe run predates head-state "
            f"saving; re-run the probe for this arm."
        )
    W, b, ln_w, ln_b, meta = [], [], [], [], None
    for p in paths:
        payload = torch.load(p, map_location="cpu", weights_only=False)
        sd = payload["state_dict"]
        if payload["head_args"]["head"] not in {"mean", "meanpool", "linear"}:
            raise ValueError(
                f"{p.name} is a {payload['head_args']['head']} head. The exact "
                f"decomposition holds only for heads linear over pooled tokens."
            )
        lin_w = next(v for k, v in sd.items() if k.endswith("weight") and v.ndim == 2)
        lin_b = next(v for k, v in sd.items() if k.endswith("bias") and v.ndim == 1
                     and v.shape[0] == lin_w.shape[0])
        norm_w = next((v for k, v in sd.items()
                       if "norm" in k.lower() and k.endswith("weight")), None)
        norm_b = next((v for k, v in sd.items()
                       if "norm" in k.lower() and k.endswith("bias")), None)
        W.append(lin_w.numpy()); b.append(lin_b.numpy())
        ln_w.append(None if norm_w is None else norm_w.numpy())
        ln_b.append(None if norm_b is None else norm_b.numpy())
        meta = payload
    return (np.mean(W, axis=0), np.mean(b, axis=0),
            {"ln_w": None if ln_w[0] is None else np.mean(ln_w, axis=0),
             "ln_b": None if ln_b[0] is None else np.mean(ln_b, axis=0),
             "config": meta["config"], "n_seeds": len(paths)})


def contributions(cache_dir: Path, W: np.ndarray, b: np.ndarray,
                  norm: dict[str, Any]) -> np.ndarray:
    """[N, 3, P] exact additive contribution of each patch to each logit.

    The LayerNorm is folded in per sample: it centres and scales the pooled
    vector, so the effective weight on a token differs by sample and cannot be
    precomputed once.
    """
    tokens = np.load(cache_dir / "tokens.npy", mmap_mode="r")
    n, n_tokens, dim = tokens.shape
    block = max(1, int(BLOCK_BYTES // max(n_tokens * dim * 4, 1)))
    out = np.empty((n, W.shape[0], n_tokens), dtype=np.float32)

    for start in range(0, n, block):
        stop = min(start + block, n)
        h = np.asarray(tokens[start:stop], dtype=np.float32)     # [B, P, D]
        pooled = h.mean(axis=1)                                   # [B, D]

        if norm["ln_w"] is not None:
            mu = pooled.mean(axis=1, keepdims=True)
            sd = pooled.std(axis=1, keepdims=True) + 1e-5
            # LayerNorm(x) = g * (x - mu) / sd + beta, so the coefficient
            # applied to each feature of every token is g / sd, per sample.
            scale = norm["ln_w"][None, :] / sd                     # [B, D]
            shift = -mu * scale + (0.0 if norm["ln_b"] is None else norm["ln_b"][None, :])
        else:
            scale = np.ones_like(pooled)
            shift = np.zeros_like(pooled)

        # contribution of patch i to logit c:
        #   (1/P) * sum_d W[c, d] * scale[b, d] * h[b, i, d]
        eff = W[None, :, :] * scale[:, None, :]                    # [B, C, D]
        out[start:stop] = np.einsum("bcd,bid->bci", eff, h) / n_tokens

        # The shift term is constant across patches and belongs to the bias, so
        # it is excluded from the per-patch map by construction. Verify that the
        # remaining sum reproduces the logit.
        logits = np.einsum("bcd,bd->bc", eff, pooled) + (shift @ W.T) + b[None, :]
        recovered = out[start:stop].sum(axis=2) + (shift @ W.T) + b[None, :]
        if not np.allclose(logits, recovered, atol=1e-3):
            raise RuntimeError(
                "Per-patch contributions do not sum to the logit. The head is "
                "not linear over pooled tokens in the way assumed."
            )
    return out


def summarise(c: np.ndarray, grid: tuple[int, ...], top_k: int) -> dict[str, Any]:
    """Concentration of positive contribution mass, per criterion."""
    out: dict[str, Any] = {}
    for j, name in enumerate(CRITERIA):
        m = np.clip(c[:, j, :], 0, None)                           # [N, P]
        total = m.sum(axis=1, keepdims=True)
        keep = total[:, 0] > 1e-9
        p = m[keep] / total[keep]
        if p.size == 0:
            out[name] = {"entropy": float("nan"), "top_k_mass": float("nan"), "n": 0}
            continue
        entropy = -(p * np.log(p + 1e-12)).sum(axis=1)
        k = min(top_k, p.shape[1])
        top = np.sort(p, axis=1)[:, -k:].sum(axis=1)
        out[name] = {
            "entropy": float(entropy.mean()),
            "entropy_sd": float(entropy.std(ddof=1)) if entropy.size > 1 else 0.0,
            "max_entropy": float(np.log(p.shape[1])),
            "top_k": k,
            "top_k_mass": float(top.mean()),
            "top_k_mass_sd": float(top.std(ddof=1)) if top.size > 1 else 0.0,
            "n": int(keep.sum()),
        }
    return out


def map_shift(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    """Cosine similarity between two arms' contribution maps, per criterion.

    Computed on the same frames in the same order, so a value near 1 means the
    two heads read the same places and a low value means the spatial pattern
    moved.
    """
    out = {}
    for j, name in enumerate(CRITERIA):
        x, y = a[:, j, :], b[:, j, :]
        num = (x * y).sum(axis=1)
        den = np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1) + 1e-12
        cos = num / den
        out[name] = {"cosine_mean": float(cos.mean()),
                     "cosine_sd": float(cos.std(ddof=1)) if cos.size > 1 else 0.0,
                     "cosine_p10": float(np.percentile(cos, 10))}
    return out


def parse_pair(spec: str) -> tuple[Path, Path]:
    probe, cache = spec.rsplit(":", 1)
    return Path(probe), Path(cache)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", required=True, type=parse_pair,
                   help="probe_dir:cache_dir for the unadapted arm")
    p.add_argument("--adapted", required=True, type=parse_pair,
                   help="probe_dir:cache_dir for the adapted arm")
    p.add_argument("--manifest", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--save-maps", action="store_true",
                   help="write the full [N, 3, P] contribution arrays for plotting")
    args = p.parse_args()

    (base_probe, base_cache), (adapt_probe, adapt_cache) = args.base, args.adapted

    ids_b, ids_a = read_index(base_cache), read_index(adapt_cache)
    if ids_b != ids_a:
        raise SystemExit(
            "The two caches hold different samples in different order. The "
            "comparison requires identical frames."
        )
    meta = pd.read_csv(args.manifest).set_index("sample_id").loc[ids_b].reset_index()

    results: dict[str, Any] = {"n_frames": len(ids_b), "arms": {}}
    maps = {}
    for label, (probe, cache) in (("base", args.base), ("adapted", args.adapted)):
        W, b, norm = load_weights(probe)
        c = contributions(cache, W, b, norm)
        maps[label] = c
        grid = json.loads((cache / "manifest.json").read_text())["encoder"]["token_layout"]["grid"]
        results["arms"][label] = {
            "probe_dir": str(probe), "grid": grid, "n_seeds": norm["n_seeds"],
            "summary": summarise(c, tuple(grid), args.top_k),
        }
        print(f"{label:<9} {probe.name}  grid {grid}  {c.shape[2]} patches")

    print(f"\nConcentration of positive contribution mass "
          f"(entropy, max {np.log(maps['base'].shape[2]):.3f})")
    print(f"{'crit':<5}{'base H':>9}{'adapt H':>10}{'change':>9}"
          f"{'base top-%d' % args.top_k:>12}{'adapt':>9}{'change':>9}")
    for c in CRITERIA:
        sb = results["arms"]["base"]["summary"][c]
        sa = results["arms"]["adapted"]["summary"][c]
        print(f"{c:<5}{sb['entropy']:>9.4f}{sa['entropy']:>10.4f}"
              f"{sa['entropy'] - sb['entropy']:>+9.4f}"
              f"{sb['top_k_mass']:>12.4f}{sa['top_k_mass']:>9.4f}"
              f"{sa['top_k_mass'] - sb['top_k_mass']:>+9.4f}")

    shift = map_shift(maps["base"], maps["adapted"])
    results["map_shift"] = shift
    print(f"\nSpatial agreement between the two arms' contribution maps")
    print(f"{'crit':<5}{'cosine':>9}{'sd':>9}{'10th pct':>10}")
    for c in CRITERIA:
        s = shift[c]
        print(f"{c:<5}{s['cosine_mean']:>9.4f}{s['cosine_sd']:>9.4f}{s['cosine_p10']:>10.4f}")
    print("  A value near 1 means the adapted head reads the same patches as the")
    print("  base head, so any performance change came from how they are weighted")
    print("  rather than from where the evidence is found.")

    # Stratified by annotator agreement, since that is where the arms differ.
    votes = {c: meta[[f"{c}_rater{r}" for r in RATERS]].to_numpy(int).sum(axis=1)
             for c in CRITERIA}
    print(f"\nSpatial agreement by annotator-agreement stratum")
    print(f"{'crit':<5}{'unanimous':>11}{'contested':>11}")
    results["map_shift_by_stratum"] = {}
    for j, c in enumerate(CRITERIA):
        un = (votes[c] == 0) | (votes[c] == 3)
        row = {}
        for name, mask in (("unanimous", un), ("contested", ~un)):
            x, y = maps["base"][mask, j, :], maps["adapted"][mask, j, :]
            num = (x * y).sum(axis=1)
            den = np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1) + 1e-12
            row[name] = float((num / den).mean())
        results["map_shift_by_stratum"][c] = row
        print(f"{c:<5}{row['unanimous']:>11.4f}{row['contested']:>11.4f}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "patch_contributions.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    if args.save_maps:
        np.savez_compressed(out_dir / "maps.npz",
                            base=maps["base"], adapted=maps["adapted"],
                            sample_ids=np.array(ids_b))
        print(f"\nmaps written to {out_dir/'maps.npz'}")
    print(f"written to {out_dir/'patch_contributions.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
