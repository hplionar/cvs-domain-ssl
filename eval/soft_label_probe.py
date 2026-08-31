#!/usr/bin/env python3
"""Do soft labels help where the problem is, or only where it never was?

SAGES Subchallenge B rewards agreement with a confidence-aware target and
reports Brier score averaged over all frames. Two very different things raise
that score: becoming better at frames where annotators disagreed, and becoming
better calibrated on frames that were already being classified correctly. A
single averaged Brier cannot separate them.

Stratified evaluation can. This script trains the same frozen-feature probe on
three target constructions and evaluates all of them against the *same* hard
majority label, split by rater agreement.

Target constructions
--------------------
hard      the binary majority vote -- what every arm in this project has used
soft      the raw vote fraction, 0, 1/3, 2/3, 1
smoothed  0.15, 0.3975, 0.645, 0.90 -- the linear smoothing used by team SDS-HD
          (2nd overall in the SAGES challenge); included so that the comparison
          is against a published recipe rather than a strawman

Evaluation
----------
Discrimination is scored against the hard majority label, unchanged, so the
numbers sit on the same axis as every other result in the project. Reported on
the full split and on the unanimous and contested strata separately. AUC is the
primary metric because the strata differ greatly in prevalence and AP moves with
prevalence; AP and its chance line are reported alongside.

Calibration is scored by Brier against the SAGES confidence-aware target

    y_conf = (1/3) * sum_i ( 0.5 + (l_i - 0.5) * c_i )

where l_i is rater i's binary label and c_i their self-reported confidence.
This is the Subchallenge B target, so the calibration column is comparable to
published numbers rather than to an invented target.

Protocol
--------
Hyperparameters are selected by GroupKFold over video_id inside the training
split, using mAP against the hard label as the criterion -- identical for every
target construction, so no arm gets a selection advantage. The validation split
is scored once, afterwards. Three seeds.

Prediction, recorded before running: if soft targets act by improving
calibration on already-easy frames, Brier will fall while contested AUC is
unchanged. If they recover signal the hard label discarded, contested AUC will
rise. The first outcome would mean the SAGES Subchallenge B gains do not
represent progress on the frames that are actually hard.

Usage:
    python eval/soft_label_probe.py \\
        --manifest metadata/sages_frames_internal_split.csv \\
        --arm dinov3_b=../cache/dinov3_b/sages \\
        --arm dinov2_b=../cache/dinov2_b/sages \\
        --arm dinov2_b_adapted=../cache/dinov2_b_adapted/sages \\
        --output-dir ../outputs/soft_label_probe
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
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

CRITERIA = ("c1", "c2", "c3")
RATERS = (1, 2, 3)

#: 0, 1/3, 2/3, 1 mapped through the SDS-HD linear smoothing (0.15 + 0.75 * v).
SMOOTHED = {0: 0.15, 1: 0.3975, 2: 0.645, 3: 0.90}

GRID = [
    {"lr": lr, "wd": wd}
    for lr in (1e-3, 3e-3, 1e-2)
    for wd in (0.0, 1e-2)
]
STEPS = 300
POOL_BLOCK_BYTES = 1 << 30


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------


def load_pooled(cache_dir: Path) -> tuple[np.ndarray, list[str]]:
    tokens = np.load(cache_dir / "tokens.npy", mmap_mode="r")
    n, n_tokens, dim = tokens.shape
    block = max(1, int(POOL_BLOCK_BYTES // max(n_tokens * dim * 4, 1)))
    pooled = np.empty((n, dim), dtype=np.float32)
    for start in range(0, n, block):
        stop = min(start + block, n)
        pooled[start:stop] = np.asarray(tokens[start:stop], dtype=np.float32).mean(axis=1)

    with open(cache_dir / "index.csv", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if [int(r["row"]) for r in rows] != list(range(len(rows))):
        raise ValueError(f"{cache_dir/'index.csv'} rows are not 0..N-1 in order.")
    if len(rows) != n:
        raise ValueError(f"{cache_dir}: index.csv {len(rows)} rows, tokens.npy {n}.")
    return pooled, [r["sample_id"] for r in rows]


def align(meta: pd.DataFrame, sample_ids: list[str]) -> pd.DataFrame:
    indexed = meta.set_index("sample_id")
    missing = [s for s in sample_ids if s not in indexed.index]
    if missing:
        raise ValueError(f"{len(missing)} sample_ids absent from manifest, e.g. {missing[:3]}")
    return indexed.loc[sample_ids].reset_index()


def vote_counts(df: pd.DataFrame) -> np.ndarray:
    """[N, 3] integer vote counts in 0..3."""
    return np.stack(
        [df[[f"{c}_rater{r}" for r in RATERS]].to_numpy(dtype=int).sum(axis=1) for c in CRITERIA],
        axis=1,
    )


def build_targets(votes: np.ndarray, kind: str) -> np.ndarray:
    if kind == "hard":
        return (votes >= 2).astype(np.float32)
    if kind == "soft":
        return (votes / 3.0).astype(np.float32)
    if kind == "smoothed":
        return np.vectorize(SMOOTHED.get)(votes).astype(np.float32)
    raise ValueError(f"Unknown target kind: {kind}")


def confidence_aware_target(df: pd.DataFrame) -> np.ndarray:
    """The SAGES Subchallenge B target, used only for scoring calibration.

    y = (1/3) sum_i ( 0.5 + (l_i - 0.5) * c_i )

    A rater with zero confidence contributes 0.5 regardless of their label; a
    fully confident rater contributes their label.
    """
    out = np.zeros((len(df), 3), dtype=np.float32)
    conf_cols = [f"confidence_rater{r}" for r in RATERS]
    missing = [c for c in conf_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Manifest lacks per-rater confidence columns: {missing}")
    conf = df[conf_cols].to_numpy(dtype=np.float32)
    for j, criterion in enumerate(CRITERIA):
        labels = df[[f"{criterion}_rater{r}" for r in RATERS]].to_numpy(dtype=np.float32)
        out[:, j] = (0.5 + (labels - 0.5) * conf).mean(axis=1)
    return out


# --------------------------------------------------------------------------
# probe
# --------------------------------------------------------------------------


def fit_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    lr: float,
    wd: float,
    seed: int,
    device: torch.device,
) -> np.ndarray:
    """Linear head on frozen features. BCEWithLogitsLoss accepts soft targets."""
    torch.manual_seed(seed)
    xt = torch.from_numpy(x_train).to(device)
    yt = torch.from_numpy(y_train).to(device)
    xe = torch.from_numpy(x_eval).to(device)

    head = nn.Sequential(nn.LayerNorm(xt.shape[1]), nn.Linear(xt.shape[1], 3)).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)
    loss_fn = nn.BCEWithLogitsLoss()

    head.train()
    for _ in range(STEPS):
        opt.zero_grad(set_to_none=True)
        loss_fn(head(xt), yt).backward()
        opt.step()
        sched.step()

    head.eval()
    with torch.no_grad():
        return torch.sigmoid(head(xe)).cpu().numpy()


def select_config(
    x: np.ndarray, y_soft: np.ndarray, y_hard: np.ndarray, groups: np.ndarray,
    seed: int, device: torch.device,
) -> dict[str, float]:
    """Grouped CV inside train. Criterion is mAP against the HARD label for
    every target construction, so no arm is selected on its own objective."""
    folds = list(GroupKFold(n_splits=3).split(x, y_hard[:, 0], groups))
    best, best_score = GRID[0], -1.0
    for config in GRID:
        scores = []
        for tr, te in folds:
            probs = fit_probe(x[tr], y_soft[tr], x[te], config["lr"], config["wd"], seed, device)
            aps = [
                average_precision_score(y_hard[te, j], probs[:, j])
                for j in range(3)
                if np.unique(y_hard[te, j]).size > 1
            ]
            if aps:
                scores.append(float(np.mean(aps)))
        score = float(np.mean(scores)) if scores else -1.0
        if score > best_score:
            best, best_score = config, score
    return best


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------


def score(probs: np.ndarray, y_hard: np.ndarray, votes: np.ndarray, y_conf: np.ndarray) -> dict[str, Any]:
    unanimous = (votes == 0) | (votes == 3)
    out: dict[str, Any] = {}
    for j, criterion in enumerate(CRITERIA):
        entry: dict[str, Any] = {}
        for name, mask in (
            ("all", np.ones(len(y_hard), dtype=bool)),
            ("unanimous", unanimous[:, j]),
            ("contested", ~unanimous[:, j]),
        ):
            y, p = y_hard[mask, j], probs[mask, j]
            if np.unique(y).size < 2:
                entry[name] = {k: float("nan") for k in ("auc", "ap", "ap_lift", "prevalence", "n")}
                continue
            prevalence = float(y.mean())
            ap = float(average_precision_score(y, p))
            entry[name] = {
                "n": int(y.size),
                "prevalence": prevalence,
                "auc": float(roc_auc_score(y, p)),
                "ap": ap,
                "ap_lift": ap - prevalence,
            }
        # Calibration against the SAGES confidence-aware target, whole split.
        entry["brier_conf"] = float(np.mean((probs[:, j] - y_conf[:, j]) ** 2))
        out[criterion] = entry
    return out


def mean_over_criteria(scored: dict[str, Any], stratum: str, metric: str) -> float:
    return float(np.nanmean([scored[c][stratum][metric] for c in CRITERIA]))


# --------------------------------------------------------------------------


def run_arm(name: str, cache_root: Path, meta: pd.DataFrame, seeds: list[int],
            device: torch.device) -> dict[str, Any]:
    x_train, ids_train = load_pooled(cache_root / "train")
    x_val, ids_val = load_pooled(cache_root / "val")
    tr, va = align(meta, ids_train), align(meta, ids_val)

    votes_tr, votes_va = vote_counts(tr), vote_counts(va)
    y_hard_tr = build_targets(votes_tr, "hard")
    y_hard_va = build_targets(votes_va, "hard")
    y_conf_va = confidence_aware_target(va)
    groups = tr["video_id"].to_numpy()

    result: dict[str, Any] = {"arm": name, "targets": {}}
    for kind in ("hard", "soft", "smoothed"):
        y_tr = build_targets(votes_tr, kind)
        per_seed, configs = [], []
        for seed in seeds:
            config = select_config(x_train, y_tr, y_hard_tr, groups, seed, device)
            configs.append(config)
            probs = fit_probe(x_train, y_tr, x_val, config["lr"], config["wd"], seed, device)
            per_seed.append(score(probs, y_hard_va, votes_va, y_conf_va))

        merged: dict[str, Any] = {}
        for criterion in CRITERIA:
            merged[criterion] = {"brier_conf": float(np.mean([s[criterion]["brier_conf"] for s in per_seed]))}
            for stratum in ("all", "unanimous", "contested"):
                merged[criterion][stratum] = {
                    metric: {
                        "mean": float(np.nanmean([s[criterion][stratum][metric] for s in per_seed])),
                        "std": float(np.nanstd([s[criterion][stratum][metric] for s in per_seed], ddof=1))
                        if len(per_seed) > 1 else 0.0,
                    }
                    for metric in ("auc", "ap", "ap_lift", "prevalence")
                }
        result["targets"][kind] = {"per_criterion": merged, "selected_configs": configs}
    return result


def report(arms: list[dict[str, Any]]) -> None:
    print(f"\nDiscrimination against the hard majority label (AUC), and calibration")
    print(f"{'arm':<20}{'target':<10}{'all':>9}{'unanimous':>11}{'contested':>11}"
          f"{'Δcont':>9}{'Brier':>9}")
    for arm in arms:
        base = None
        for kind in ("hard", "soft", "smoothed"):
            m = arm["targets"][kind]["per_criterion"]
            a = float(np.mean([m[c]["all"]["auc"]["mean"] for c in CRITERIA]))
            u = float(np.mean([m[c]["unanimous"]["auc"]["mean"] for c in CRITERIA]))
            k = float(np.mean([m[c]["contested"]["auc"]["mean"] for c in CRITERIA]))
            b = float(np.mean([m[c]["brier_conf"] for c in CRITERIA]))
            if kind == "hard":
                base = k
            delta = k - base
            print(f"{arm['arm']:<20}{kind:<10}{a:>9.4f}{u:>11.4f}{k:>11.4f}"
                  f"{delta:>+9.4f}{b:>9.4f}")
    print("  Δcont = change in contested-stratum AUC relative to the hard-label arm.")
    print("  Brier is against the SAGES confidence-aware target; lower is better.")
    print("  Soft labels that lower Brier without raising Δcont improved calibration")
    print("  on frames that were never the difficulty.")


def parse_arm(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError(f"--arm must be name=cache_root, got {spec!r}")
    name, root = spec.split("=", 1)
    return name, Path(root)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True)
    p.add_argument("--arm", action="append", required=True, type=parse_arm,
                   help="name=cache_root, where cache_root contains train/ and val/")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    meta = pd.read_csv(args.manifest)
    device = torch.device(args.device)
    seeds = list(range(args.seeds))

    arms = []
    for name, root in args.arm:
        print(f"running {name} ({root})")
        try:
            arms.append(run_arm(name, root, meta, seeds, device))
        except (FileNotFoundError, ValueError) as exc:
            print(f"[skipped] {name}: {exc}")
    if not arms:
        return 1

    report(arms)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "soft_label_probe.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"seeds": seeds, "grid": GRID, "steps": STEPS, "arms": arms}, fh, indent=2)
    print(f"\nwritten to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
