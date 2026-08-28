#!/usr/bin/env python3
"""Score a held-out test cache using the configuration already selected on val.

Every figure in the frozen-probe ranking is a validation figure, selected as the
maximum over 24 configurations of the mean over seeds of the maximum over epochs
of validation mAP. Two maxima taken on the split then reported. This script
produces the held-out figure instead.

Nothing is selected here. The configuration, the seed list, the epoch budget and
the stopping epoch are all read from the ``results.json`` that the probe already
wrote. The head is retrained with those exact settings -- seconds on cached
features -- and applied to the test cache once.

**Self-verification.** Retraining only reproduces the original run if every
setting matches, including the batch size, which ``results.json`` does not
record. So before any test number is reported, the reproduced validation mAP at
the selected epoch is compared against the value stored in
``history_seed<N>.json``. A mismatch beyond ``--tolerance`` aborts: a test score
from a head that is not the selected head is worse than no test score.

The test cache is checked against the train cache by the same protocol
verification the probe uses, so a cache built with a different preprocessing or
reduction is refused rather than silently scored.

Usage:
    python eval/score_test_cache.py \\
        --probe-dir ../outputs/cvs-domain-ssl/probe/dinov3_b_sages_mean \\
        --train-features ../cache/dinov3_b/sages/train \\
        --val-features   ../cache/dinov3_b/sages/val \\
        --test-features  ../cache/dinov3_b/sages/test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from eval.metrics import compute_multilabel_metrics_from_logits
from models.heads.attentive_head import build_head
from train.train_probe_cached import (
    CachedFeatures,
    PooledFeatures,
    compute_pos_weight,
    set_seed,
    verify_protocol,
    verify_same_encoder,
)

PRECOMPUTE_HEADS = {"mean", "meanpool", "linear"}


@torch.no_grad()
def infer(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits, targets = [], []
    for tokens, target in loader:
        logits.append(model(tokens.to(device, non_blocking=True)).logits.float().cpu().numpy())
        targets.append(target.numpy())
    return np.concatenate(logits), np.concatenate(targets)


def retrain_to_epoch(
    config: dict[str, float],
    seed: int,
    stop_epoch: int,
    total_epochs: int,
    head_kind: str,
    head_kwargs: dict[str, Any],
    feature_dim: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    pos_weight: torch.Tensor | None,
    device: torch.device,
) -> tuple[nn.Module, dict[str, float]]:
    """Reproduce one selected run, stopping at the epoch it was selected at.

    ``T_max`` for the scheduler is the *configured* epoch budget, not the
    stopping epoch, because that is what the original run used. Getting this
    wrong changes the learning-rate path and the verification will catch it.
    """
    set_seed(seed)
    head = build_head(head_kind, feature_dim, dropout=float(config["dropout"]), **head_kwargs).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=float(config["lr"]), weight_decay=float(config["weight_decay"])
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)
    criterion = nn.BCEWithLogitsLoss(pos_weight=None if pos_weight is None else pos_weight.to(device))

    metrics: dict[str, float] = {}
    for epoch in range(1, stop_epoch + 1):
        head.train()
        for tokens, target in train_loader:
            tokens = tokens.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            loss = criterion(head(tokens).logits, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        scheduler.step()
        logits, targets = infer(head, val_loader, device)
        metrics = compute_multilabel_metrics_from_logits(targets, logits)
    return head, metrics


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--probe-dir", required=True, help="directory containing results.json")
    p.add_argument("--train-features", required=True)
    p.add_argument("--val-features", required=True)
    p.add_argument("--test-features", required=True)
    p.add_argument("--batch-size", type=int, default=256,
                   help="must match the original run; results.json does not record it")
    p.add_argument("--tolerance", type=float, default=1e-4,
                   help="allowed |reproduced - recorded| validation mAP")
    p.add_argument("--output-name", default="test_metrics.json")
    p.add_argument("--device", default="cuda")
    p.add_argument("--in-memory", action="store_true", default=None)
    p.add_argument("--allow-mismatch", action="store_true",
                   help="report anyway if reproduction fails; the result is then not the selected head")
    args = p.parse_args()

    probe_dir = Path(args.probe_dir)
    summary = json.loads((probe_dir / "results.json").read_text())
    config = summary["selected"]["config"]
    seeds = summary["search"]["seeds"]
    total_epochs = summary["search"]["epochs"]
    use_pos_weight = bool(summary["search"].get("pos_weight", False))
    head_kind = summary["head"]["kind"]
    head_kwargs = {
        "hidden_dim": summary["head"].get("hidden_dim", 128),
        "num_branches": summary["head"].get("branches", 1),
    }

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    train_cache = CachedFeatures(args.train_features, in_memory=args.in_memory)
    val_cache = CachedFeatures(args.val_features, in_memory=args.in_memory)
    test_cache = CachedFeatures(args.test_features, in_memory=args.in_memory)

    verify_same_encoder(train_cache.manifest, val_cache.manifest)
    verify_same_encoder(train_cache.manifest, test_cache.manifest)
    verify_protocol(train_cache.manifest, val_cache.manifest, label_a="train", label_b="val")
    verify_protocol(train_cache.manifest, test_cache.manifest, label_a="train", label_b="test")

    precompute = head_kind in PRECOMPUTE_HEADS
    if precompute:
        train_ds: Dataset = PooledFeatures(train_cache.pooled(), train_cache.all_targets())
        val_ds: Dataset = PooledFeatures(val_cache.pooled(), val_cache.all_targets())
        test_ds: Dataset = PooledFeatures(test_cache.pooled(), test_cache.all_targets())
    else:
        train_ds, val_ds, test_ds = train_cache, val_cache, test_cache

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    pos_weight = compute_pos_weight(train_cache.all_targets()) if use_pos_weight else None

    print(f"encoder      {train_cache.manifest['encoder']['checkpoint_id']}")
    print(f"head         {head_kind}   config {config}")
    print(f"seeds        {seeds}   epoch budget {total_epochs}   pos_weight {use_pos_weight}")
    print(f"test         {len(test_ds)} samples\n")

    per_seed, failures = [], []
    for seed in seeds:
        history_path = probe_dir / f"history_seed{seed}.json"
        if not history_path.is_file():
            failures.append(f"seed {seed}: no {history_path.name}")
            continue
        history = json.loads(history_path.read_text())
        # The probe selected the epoch with the highest validation mAP.
        recorded = max(history, key=lambda row: row["mAP"])
        stop_epoch, recorded_map = int(recorded["epoch"]), float(recorded["mAP"])

        head, val_metrics = retrain_to_epoch(
            config, seed, stop_epoch, total_epochs, head_kind, head_kwargs,
            train_cache.feature_dim, train_loader, val_loader, pos_weight, device,
        )
        delta = abs(val_metrics["mAP"] - recorded_map)
        ok = delta <= args.tolerance
        flag = "ok" if ok else "MISMATCH"
        print(f"seed {seed}  epoch {stop_epoch:>3}  val mAP recorded {recorded_map:.6f} "
              f"reproduced {val_metrics['mAP']:.6f}  delta {delta:.2e}  {flag}")
        if not ok:
            failures.append(f"seed {seed}: val mAP delta {delta:.2e} exceeds {args.tolerance}")
            if not args.allow_mismatch:
                continue

        logits, targets = infer(head, test_loader, device)
        np.savez(probe_dir / f"test_logits_seed{seed}.npz", logits=logits, targets=targets)
        test_metrics = compute_multilabel_metrics_from_logits(targets, logits)
        per_seed.append({
            "seed": seed,
            "stop_epoch": stop_epoch,
            "val_map_recorded": recorded_map,
            "val_map_reproduced": float(val_metrics["mAP"]),
            "reproduction_delta": float(delta),
            "test": {k: float(v) for k, v in test_metrics.items()},
        })

    if failures:
        print("\nreproduction problems:")
        for f in failures:
            print(f"  {f}")
        if not args.allow_mismatch:
            print("\nNo test scores written for the affected seeds. Most likely cause is a\n"
                  "--batch-size that differs from the original run. Try 32, 64 or 128.")
    if not per_seed:
        return 1

    keys = sorted(per_seed[0]["test"])
    aggregate = {
        k: {
            "mean": float(np.nanmean([s["test"][k] for s in per_seed])),
            "std": float(np.nanstd([s["test"][k] for s in per_seed], ddof=1)) if len(per_seed) > 1 else 0.0,
        }
        for k in keys
    }

    print(f"\n{'metric':<12}{'val (selected)':>16}{'test':>10}{'+/-':>9}")
    val_map = summary["selected"]["mean_map"]
    print(f"{'mAP':<12}{val_map:>16.4f}{aggregate['mAP']['mean']:>10.4f}{aggregate['mAP']['std']:>9.4f}")
    for k in ("mean_auc", "mean_bacc", "c1_ap", "c2_ap", "c3_ap"):
        if k in aggregate:
            print(f"{k:<12}{'':>16}{aggregate[k]['mean']:>10.4f}{aggregate[k]['std']:>9.4f}")
    print(f"\nselection-to-test drop in mAP: {aggregate['mAP']['mean'] - val_map:+.4f}")

    payload = {
        "probe_dir": str(probe_dir),
        "encoder": train_cache.manifest["encoder"],
        "test_cache": str(args.test_features),
        "n_test": len(test_ds),
        "selected_config": config,
        "batch_size": args.batch_size,
        "val_map_selected": val_map,
        "test": aggregate,
        "per_seed": per_seed,
        "reproduction_failures": failures,
    }
    out = probe_dir / args.output_name
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"written to {out}")
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
