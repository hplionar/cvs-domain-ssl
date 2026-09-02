#!/usr/bin/env python3
"""Score a held-out test cache using the head that was selected on validation.

Every figure in the frozen-probe ranking is a validation figure, chosen as the
maximum over 24 configurations of the mean over seeds of the maximum over epochs
of validation mAP -- two maxima taken on the split that is then reported. This
script produces the held-out figure instead.

Nothing is selected here and nothing is retrained. The parameters saved at the
selected epoch by ``train_probe_cached.py`` are loaded and applied to the test
cache once. An earlier version reconstructed the head from the recorded
configuration; it did not reproduce the recorded validation mAP at any batch
size tried, because ``build_head_for`` also passes branch count and global
source. Loading the saved parameters removes that dependence entirely.

The validation mAP is nonetheless recomputed as a check. If the loaded head
does not reproduce the value recorded when it was saved, then the head and the
caches do not correspond -- the wrong arm, or the wrong probe directory -- and
no test score is written.

Requires probe runs made after the head-state patch. Runs predating it have no
``head_seed*.pt`` and must be re-run; on cached features with ``--in-memory``
this takes minutes per arm.

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
from train.train_probe_cached import (
    _prefix_to_device,
    build_head_for,
    CachedFeatures,
    PooledFeatures,
    verify_protocol,
    verify_same_encoder,
)

PRECOMPUTE_HEADS = {"mean", "meanpool", "linear"}


@torch.no_grad()
def infer(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits, targets = [], []
    for tokens, prefix, target in loader:
        out = model(tokens.to(device, non_blocking=True), _prefix_to_device(prefix, device))
        logits.append(out.logits.float().cpu().numpy())
        targets.append(target.numpy())
    return np.concatenate(logits), np.concatenate(targets)


def build_from_checkpoint(payload: dict[str, Any], feature_dim: int,
                          device: torch.device) -> nn.Module:
    """Rebuild the head through the trainer's own construction site.

    ``build_head_for`` is the one place the trainer constructs a head, and it
    reads ``hidden_dim`` from the searched configuration while taking branch
    count and global source from the arguments. Reconstructing those arguments
    here independently is what caused an earlier version of this script to build
    a subtly different head; calling the same function with the saved values
    removes the possibility.
    """
    saved_dim = payload.get("feature_dim")
    if saved_dim is not None and int(saved_dim) != int(feature_dim):
        raise ValueError(
            f"Head was trained on {saved_dim}-dimensional features but the cache "
            f"supplies {feature_dim}. Wrong arm, or the wrong cache."
        )
    head_args = argparse.Namespace(**payload["head_args"])
    head = build_head_for(payload["config"], feature_dim, head_args).to(device)
    head.load_state_dict(payload["state_dict"])
    return head


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--probe-dir", required=True)
    p.add_argument("--train-features", required=True,
                   help="used only for protocol verification against the test cache")
    p.add_argument("--val-features", required=True)
    p.add_argument("--test-features", required=True)
    p.add_argument("--batch-size", type=int, default=256,
                   help="inference only; does not affect the result")
    p.add_argument("--tolerance", type=float, default=1e-4,
                   help="allowed |recomputed - saved| validation mAP")
    p.add_argument("--output-name", default="test_metrics.json")
    p.add_argument("--logits-prefix", default="test_logits",
                   help="filename stem for the saved logits. Two evaluations of "
                        "the same arm against different test caches would "
                        "otherwise collide, silently replacing one set with the "
                        "other while leaving both metrics files in place.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--in-memory", action="store_true", default=None)
    args = p.parse_args()

    probe_dir = Path(args.probe_dir)
    checkpoints = sorted(probe_dir.glob("head_seed*.pt"))
    if not checkpoints:
        print(f"No head_seed*.pt in {probe_dir}.\n"
              f"This probe run predates the head-state patch; re-run the probe "
              f"for this arm.")
        return 1

    summary = json.loads((probe_dir / "results.json").read_text())
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_cache = CachedFeatures(args.train_features, in_memory=args.in_memory)
    val_cache = CachedFeatures(args.val_features, in_memory=args.in_memory)
    test_cache = CachedFeatures(args.test_features, in_memory=args.in_memory)

    verify_same_encoder(train_cache.manifest, val_cache.manifest)
    verify_same_encoder(train_cache.manifest, test_cache.manifest)
    verify_protocol(train_cache.manifest, val_cache.manifest, label_a="train", label_b="val")
    verify_protocol(train_cache.manifest, test_cache.manifest, label_a="train", label_b="test")

    head_kind = torch.load(checkpoints[0], map_location="cpu",
                           weights_only=False)["head_args"]["head"]
    precompute = head_kind in PRECOMPUTE_HEADS
    if precompute:
        val_ds: Dataset = PooledFeatures(val_cache.pooled(), val_cache.all_targets())
        test_ds: Dataset = PooledFeatures(test_cache.pooled(), test_cache.all_targets())
    else:
        val_ds, test_ds = val_cache, test_cache

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"encoder      {train_cache.manifest['encoder']['checkpoint_id']}")
    print(f"head         {head_kind}   config {summary['selected']['config']}")
    print(f"checkpoints  {len(checkpoints)}   test {len(test_ds)} samples\n")

    per_seed, failures = [], []
    for path in checkpoints:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        seed = int(payload["seed"])
        head = build_from_checkpoint(payload, train_cache.feature_dim, device)

        val_logits, val_targets = infer(head, val_loader, device)
        recomputed = float(compute_multilabel_metrics_from_logits(val_targets, val_logits)["mAP"])
        saved = float(payload["val_map"])
        delta = abs(recomputed - saved)
        ok = delta <= args.tolerance
        print(f"seed {seed}  epoch {int(payload['epoch']):>3}  val mAP saved {saved:.6f} "
              f"recomputed {recomputed:.6f}  delta {delta:.2e}  {'ok' if ok else 'MISMATCH'}")
        if not ok:
            failures.append(f"seed {seed}: val mAP delta {delta:.2e} exceeds {args.tolerance}")
            continue

        logits, targets = infer(head, test_loader, device)
        np.savez(probe_dir / f"{args.logits_prefix}_seed{seed}.npz",
                 logits=logits, targets=targets)
        metrics = compute_multilabel_metrics_from_logits(targets, logits)
        per_seed.append({
            "seed": seed,
            "epoch": int(payload["epoch"]),
            "val_map_saved": saved,
            "val_map_recomputed": recomputed,
            "test": {k: float(v) for k, v in metrics.items()},
        })

    if failures:
        print("\nverification problems:")
        for f in failures:
            print(f"  {f}")
        print("The loaded head does not reproduce the validation score recorded when it\n"
              "was saved, so the head and the caches do not correspond. Check that the\n"
              "probe directory and the feature caches are the same arm.")
    if not per_seed:
        return 1

    aggregate = {
        k: {
            "mean": float(np.nanmean([s["test"][k] for s in per_seed])),
            "std": float(np.nanstd([s["test"][k] for s in per_seed], ddof=1))
            if len(per_seed) > 1 else 0.0,
        }
        for k in sorted(per_seed[0]["test"])
    }

    val_map = summary["selected"]["mean_map"]
    print(f"\n{'metric':<12}{'val (selected)':>16}{'test':>10}{'+/-':>9}")
    print(f"{'mAP':<12}{val_map:>16.4f}{aggregate['mAP']['mean']:>10.4f}"
          f"{aggregate['mAP']['std']:>9.4f}")
    for k in ("mean_auc", "mean_bacc", "c1_ap", "c2_ap", "c3_ap"):
        if k in aggregate:
            print(f"{k:<12}{'':>16}{aggregate[k]['mean']:>10.4f}{aggregate[k]['std']:>9.4f}")
    print(f"\nselection-to-test difference in mAP: "
          f"{aggregate['mAP']['mean'] - val_map:+.4f}")

    payload_out = {
        "probe_dir": str(probe_dir),
        "encoder": train_cache.manifest["encoder"],
        "test_cache": str(args.test_features),
        "n_test": len(test_ds),
        "selected_config": summary["selected"]["config"],
        "val_map_selected": val_map,
        "test": aggregate,
        "per_seed": per_seed,
        "verification_failures": failures,
    }
    out = probe_dir / args.output_name
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(payload_out, fh, indent=2)
    print(f"written to {out}")
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
