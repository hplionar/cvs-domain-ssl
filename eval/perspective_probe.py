#!/usr/bin/env python3
"""Predicting each annotator's judgement rather than their consensus.

Endoscapes releases the individual judgements of three annotators alongside the
majority label, and every published method for this task discards two of the
three. Section 5.20 of the notes shows what that costs: the three apply
thresholds differing by up to eightfold and differing by criterion, so that on
C1 the consensus of three is in practice the agreement of two.

The learning-with-disagreements literature -- Uma et al.'s survey, the three
LeWiDi shared tasks, and the perspectivist evaluation introduced in the 2025
edition -- argues that reconciling divergent annotations misrepresents the
evidence, and that a model should predict the distribution or the individual
judgements rather than a reconciled label. That framing has not been applied to
a surgical safety criterion.

**Why this is not the soft-label experiment again.** Training on vote fractions
improved calibration and left discrimination on contested frames unchanged, and
the reason is structural: a head with one output per criterion can only shift
its confidence, and cannot represent which annotator dissented. Nine outputs
can. Whether the encoder's features support that is the question.

**What is predicted.** Three criteria by three annotators, from the same pooled
frozen features and under the same protocol as the consensus probe, so a
difference between the two is attributable to the target and not to the model.

**What is reported.**

    per annotator   average precision for each of the nine outputs, against
                    that annotator's own labels. An annotator whose positive
                    rate is 3.7% will be predicted near-constant, and the
                    average precision will say so.
    consensus       the nine outputs thresholded and voted, scored against the
                    majority label, which is comparable to the consensus probe
                    and to published figures.
    disagreement    whether the three heads disagreeing predicts the three
                    annotators disagreeing. This costs nothing once the model
                    exists and asks whether ambiguity is anticipable from the
                    frame, which Section 5.16 could not settle.

Usage:
    python eval/perspective_probe.py \\
        --train-features ../cache/dinov3_b/endoscapes/train \\
        --val-features   ../cache/dinov3_b/endoscapes/val \\
        --test-features  ../cache/dinov3_b/endoscapes/test \\
        --manifest metadata/endoscapes_frames.csv \\
        --output-dir ../outputs/perspective/dinov3_b
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from train.train_probe_cached import CachedFeatures, set_seed

CRITERIA = ("c1", "c2", "c3")
ANNOTATORS = (1, 2, 3)


def read_index(cache: Path) -> list[str]:
    with open(cache / "index.csv", newline="", encoding="utf-8") as fh:
        return [r["sample_id"] for r in csv.DictReader(fh)]


def per_annotator_targets(cache: Path, meta: pd.DataFrame) -> torch.Tensor:
    """[N, 3 annotators, 3 criteria] from the manifest's per-annotator columns.

    The columns hold a stringified list of three binary judgements, one per
    criterion, so they are parsed rather than read as numbers. Rows are taken in
    the cache's order, since the features are in that order and a mismatch would
    be silent.
    """
    ids = read_index(cache)
    indexed = meta.set_index("sample_id")
    missing = [s for s in ids if s not in indexed.index]
    if missing:
        raise SystemExit(f"{len(missing)} sample_ids absent from the manifest, "
                         f"e.g. {missing[:3]}")
    rows = indexed.loc[ids]

    out = np.zeros((len(ids), len(ANNOTATORS), len(CRITERIA)), dtype=np.float32)
    for a, annotator in enumerate(ANNOTATORS):
        column = rows[f"cvs_annotator_{annotator}"]
        if column.isna().any():
            raise SystemExit(
                f"cvs_annotator_{annotator} is missing for "
                f"{int(column.isna().sum())} of {len(rows)} cached frames. "
                f"Filter the cache to annotated frames before probing."
            )
        parsed = column.apply(ast.literal_eval)
        for i, value in enumerate(parsed):
            out[i, a] = value
    return torch.from_numpy(out)


class PerspectiveHead(nn.Module):
    """LayerNorm and a linear map to nine outputs, reshaped to annotator by
    criterion.

    Deliberately the consensus head with a wider output layer. Anything more
    expressive would confound the target with the architecture, and the claim
    being tested is about what the features support rather than about a better
    head.
    """

    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(dim, len(ANNOTATORS) * len(CRITERIA))

    def forward(self, x):
        y = self.fc(self.drop(self.norm(x)))
        return y.view(-1, len(ANNOTATORS), len(CRITERIA))


@torch.no_grad()
def infer(head, loader, device):
    head.eval()
    logits, targets = [], []
    for x, y in loader:
        logits.append(head(x.to(device)).float().cpu().numpy())
        targets.append(y.numpy())
    return np.concatenate(logits), np.concatenate(targets)


def mean_ap(logits: np.ndarray, targets: np.ndarray) -> float:
    """Mean average precision over the nine outputs, skipping degenerate ones.

    An annotator with no positives in a split yields an undefined average
    precision. Skipping rather than substituting zero keeps the mean
    interpretable, and the count of skipped outputs is reported.
    """
    scores = []
    for a in range(len(ANNOTATORS)):
        for c in range(len(CRITERIA)):
            y = targets[:, a, c]
            if np.unique(y).size < 2:
                continue
            scores.append(average_precision_score(y, logits[:, a, c]))
    return float(np.mean(scores)) if scores else float("nan")


def train_one(config, seed, train_ds, val_ds, dim, args, device):
    set_seed(seed)
    head = PerspectiveHead(dim, float(config["dropout"])).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=float(config["lr"]),
                            weight_decay=float(config["weight_decay"]))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.BCEWithLogitsLoss()

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)

    best, best_state, left = -1.0, None, args.patience
    for _ in range(args.epochs):
        head.train()
        for x, y in train_loader:
            loss = loss_fn(head(x.to(device)), y.to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sched.step()
        logits, targets = infer(head, val_loader, device)
        current = mean_ap(logits, targets)
        if not np.isnan(current) and current > best:
            best = current
            best_state = {k: v.detach().cpu().clone()
                          for k, v in head.state_dict().items()}
            left = args.patience
        else:
            left -= 1
            if args.patience > 0 and left <= 0:
                break
    return best, best_state


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-features", required=True)
    p.add_argument("--val-features", required=True)
    p.add_argument("--test-features", default=None)
    p.add_argument("--manifest", required=True)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    meta = pd.read_csv(args.manifest)

    caches = {"train": Path(args.train_features), "val": Path(args.val_features)}
    if args.test_features:
        caches["test"] = Path(args.test_features)

    features, targets = {}, {}
    for split, path in caches.items():
        cache = CachedFeatures(path)
        features[split] = cache.pooled()
        targets[split] = per_annotator_targets(path, meta)
    dim = features["train"].shape[1]

    encoder = CachedFeatures(caches["train"]).manifest["encoder"]["checkpoint_id"]
    print(f"encoder     {encoder}")
    print(f"features    {dim}-dim, " +
          ", ".join(f"{len(features[s])} {s}" for s in features))

    print(f"\npositive rate per annotator and criterion, training split")
    print(f"{'annotator':<11}" + "".join(f"{c.upper():>9}" for c in CRITERIA))
    y = targets["train"].numpy()
    for a, annotator in enumerate(ANNOTATORS):
        print(f"{annotator:<11}" + "".join(f"{y[:, a, c].mean():>9.3f}"
                                           for c in range(len(CRITERIA))))
    print("  An annotator whose rate is near zero will be predicted")
    print("  near-constant, and the average precision below will show it.")

    grid = [{"lr": lr, "weight_decay": wd, "dropout": dr}
            for lr in (1e-3, 3e-3) for wd in (0.0, 1e-2) for dr in (0.0, 0.1)]
    train_ds = TensorDataset(features["train"], targets["train"])
    val_ds = TensorDataset(features["val"], targets["val"])

    best_cfg, best_score, best_states = None, -1.0, None
    for cfg in grid:
        runs = [train_one(cfg, s, train_ds, val_ds, dim, args, device)
                for s in range(args.seeds)]
        mean = float(np.mean([r[0] for r in runs]))
        if mean > best_score:
            best_cfg, best_score, best_states = cfg, mean, [r[1] for r in runs]

    print(f"\nselected {best_cfg}, validation mean AP over nine outputs "
          f"{best_score:.4f}")

    results: dict[str, Any] = {"encoder": encoder, "config": best_cfg,
                               "val_map": best_score}

    split = "test" if "test" in features else "val"
    loader = DataLoader(TensorDataset(features[split], targets[split]),
                        batch_size=512, shuffle=False)
    per_seed = []
    for state in best_states:
        head = PerspectiveHead(dim, float(best_cfg["dropout"])).to(device)
        head.load_state_dict(state)
        per_seed.append(infer(head, loader, device))
    logits = np.mean([s[0] for s in per_seed], axis=0)
    truth = per_seed[0][1]

    print(f"\naverage precision per output, {split} split")
    print(f"{'annotator':<11}" + "".join(f"{c.upper():>9}" for c in CRITERIA)
          + f"{'prevalence':>26}")
    results["per_annotator"] = {}
    for a, annotator in enumerate(ANNOTATORS):
        aps, prevs = [], []
        for c in range(len(CRITERIA)):
            yc = truth[:, a, c]
            ap = (average_precision_score(yc, logits[:, a, c])
                  if np.unique(yc).size > 1 else float("nan"))
            aps.append(ap); prevs.append(float(yc.mean()))
        results["per_annotator"][annotator] = {
            "ap": aps, "prevalence": prevs}
        print(f"{annotator:<11}" + "".join(f"{v:>9.4f}" for v in aps)
              + "   " + " ".join(f"{v:.3f}" for v in prevs))

    # Consensus, for comparability with the ordinary probe and with published
    # figures. The nine outputs are thresholded at zero and voted, which is the
    # rule the dataset itself uses to form the label.
    votes = (logits > 0).sum(axis=1)
    consensus_truth = (truth.sum(axis=1) >= 2).astype(int)
    print(f"\nconsensus, formed by voting the three predicted judgements")
    print(f"{'criterion':<11}{'AP':>9}{'AUC':>9}{'prevalence':>12}")
    results["consensus"] = {}
    for c, name in enumerate(CRITERIA):
        yc = consensus_truth[:, c]
        score = logits[:, :, c].mean(axis=1)
        ap = average_precision_score(yc, score) if np.unique(yc).size > 1 else float("nan")
        auc = roc_auc_score(yc, score) if np.unique(yc).size > 1 else float("nan")
        results["consensus"][name] = {"ap": ap, "auc": auc, "prevalence": float(yc.mean())}
        print(f"{name.upper():<11}{ap:>9.4f}{auc:>9.4f}{yc.mean():>12.3f}")
    mean_consensus_ap = float(np.nanmean([results["consensus"][c]["ap"] for c in CRITERIA]))
    print(f"{'mean':<11}{mean_consensus_ap:>9.4f}")
    results["consensus_map"] = mean_consensus_ap

    # Does the model's own disagreement predict the annotators'? Free, given the
    # model, and it asks what Section 5.16 could not: whether ambiguity is
    # anticipable from the frame.
    print(f"\ndoes predicted disagreement match observed disagreement?")
    print(f"{'criterion':<11}{'AUC':>9}{'contested':>12}")
    results["disagreement"] = {}
    for c, name in enumerate(CRITERIA):
        contested = ((truth[:, :, c].sum(axis=1) == 1) |
                     (truth[:, :, c].sum(axis=1) == 2)).astype(int)
        # Spread among the three predicted logits: large where the heads split.
        spread = logits[:, :, c].std(axis=1)
        auc = (roc_auc_score(contested, spread)
               if np.unique(contested).size > 1 else float("nan"))
        results["disagreement"][name] = {"auc": auc,
                                         "contested_rate": float(contested.mean())}
        print(f"{name.upper():<11}{auc:>9.4f}{contested.mean():>12.3f}")
    print("  An AUC near 0.5 means disagreement is not anticipable from the")
    print("  frame, which would be consistent with the null in Section 5.19.")
    print("  Well above 0.5 would mean the model can flag frames on which")
    print("  annotators are likely to split, which is a clinically useful")
    print("  output the consensus formulation cannot produce.")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "perspective_probe.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwritten to {out / 'perspective_probe.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
