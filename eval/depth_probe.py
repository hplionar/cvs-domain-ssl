#!/usr/bin/env python3
"""Where in the encoder does the criterion information live?

Every result in this project reads the final block. That is convention rather
than measurement: for tasks defined over spatial arrangement, intermediate
layers often carry more, and the semantic-diffusion account predicts exactly
that here. If global information diffuses into patch tokens with depth, then
earlier layers should retain more within-frame differentiation and might
support the task better despite being less semantically abstract.

The extraction writes ``layers.npy`` at ``[N, depths, patches, dim]``, which
``train_probe_cached.py`` cannot read: it expects ``[N, patches, dim]`` and has
no notion of a layer axis. This script slices one depth at a time and fits the
same head under the same protocol, so the only thing varying across the reported
rows is which block the features come from.

Two questions in one pass:

    is the last layer the right place to read?   If an intermediate depth
        outperforms depth 1.0, the standard practice is wrong for this task and
        the margin says by how much.

    where does adaptation change the encoder?   Running a base and an adapted
        arm gives a curve each. If they coincide early and separate late,
        continued pretraining altered the upper blocks and left the lower ones;
        if they separate from the first depth measured, it altered the whole
        stack.

**A caveat that belongs in the write-up.** Each depth is normalised by the
encoder's terminal LayerNorm before caching, which is what makes depth 1.0
reproduce the tensor every other experiment used. Applying a normalisation
fitted for the last block to the fourth is not neutral. The alternative --
comparing an unnormalised early layer against a normalised final one -- confounds
depth with feature scale, which is worse, but the choice should be stated rather
than assumed.

Usage:
    python eval/depth_probe.py \\
        --arm dinov3_b --cache-root ../cache_depth \\
        --output-dir ../outputs/depth_probe
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
from train.train_probe_cached import build_head_for, set_seed


class DepthFeatures(Dataset):
    """One layer of a depth cache, mean-pooled, in the head's expected shape.

    The head is fitted over pooled tokens, so the patch grid is collapsed here
    rather than inside the head. That makes each depth exactly the frozen-probe
    configuration with different features, which is what allows the comparison
    across depths to be attributed to depth.
    """

    def __init__(self, pooled: torch.Tensor, targets: torch.Tensor) -> None:
        self.pooled = pooled
        self.targets = targets

    def __len__(self) -> int:
        return self.pooled.shape[0]

    def __getitem__(self, i: int):
        empty = torch.zeros(0, self.pooled.shape[-1], dtype=self.pooled.dtype)
        return self.pooled[i].unsqueeze(0), empty, self.targets[i]


def pooled_at_depth(cache: Path, index: int, block: int = 256) -> torch.Tensor:
    """Mean over patches at one depth, read in blocks.

    A depth cache is four times the size of an ordinary one, so the whole array
    is not brought into memory; only the requested layer is, and in pieces.
    """
    layers = np.load(cache / "layers.npy", mmap_mode="r")
    n = layers.shape[0]
    out = torch.empty(n, layers.shape[3], dtype=torch.float32)
    for start in range(0, n, block):
        stop = min(start + block, n)
        chunk = np.asarray(layers[start:stop, index], dtype=np.float32)
        out[start:stop] = torch.from_numpy(chunk.mean(axis=1))
    return out


@torch.no_grad()
def infer(head: nn.Module, loader: DataLoader, device: torch.device):
    head.eval()
    logits, targets = [], []
    for tokens, prefix, target in loader:
        out = head(tokens.to(device), None)
        logits.append(out.logits.float().cpu().numpy())
        targets.append(target.numpy())
    return np.concatenate(logits), np.concatenate(targets)


def fit_one(config: dict[str, float], seed: int, train_ds, val_ds, dim: int,
            args, device) -> tuple[float, int]:
    set_seed(seed)
    head_args = argparse.Namespace(head="mean", attn_branches=1,
                                   global_source="auto")
    head = build_head_for(config, dim, head_args).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=float(config["lr"]),
                            weight_decay=float(config["weight_decay"]))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.BCEWithLogitsLoss()

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=0)

    best, best_epoch, left = -1.0, 0, args.patience
    for epoch in range(1, args.epochs + 1):
        head.train()
        for tokens, prefix, target in train_loader:
            tokens = tokens.to(device)
            target = target.to(device)
            loss = loss_fn(head(tokens, None).logits, target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sched.step()
        logits, targets = infer(head, val_loader, device)
        current = compute_multilabel_metrics_from_logits(targets, logits)["mAP"]
        if not np.isnan(current) and current > best:
            best, best_epoch, left = float(current), epoch, args.patience
        else:
            left -= 1
            if args.patience > 0 and left <= 0:
                break
    return best, best_epoch


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", action="append", required=True,
                   help="cache directory name under --cache-root (repeatable, "
                        "so a base and an adapted arm can be compared)")
    p.add_argument("--cache-root", default="../cache_depth")
    p.add_argument("--dataset", default="sages")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    grid = [{"lr": lr, "weight_decay": wd, "dropout": dr}
            for lr in (1e-3, 3e-3) for wd in (0.0, 1e-2) for dr in (0.0, 0.1)]

    results: dict[str, Any] = {"arms": {}}
    root = Path(args.cache_root)

    for arm in args.arm:
        train_c = root / arm / args.dataset / "train"
        val_c = root / arm / args.dataset / "val"
        for c in (train_c, val_c):
            if not (c / "layers.npy").is_file():
                raise SystemExit(f"No layers.npy in {c}. Re-extract with "
                                 f"--layer-depths.")
        manifest = json.loads((val_c / "manifest.json").read_text())
        depths = manifest["layer_depths"]
        shape = manifest["shapes"]["layers"]

        print(f"\n{arm}   {manifest['encoder']['checkpoint_id']}")
        print(f"  depths {depths}   layers {shape}")

        train_y = torch.from_numpy(np.load(train_c / "targets.npy")).float()
        val_y = torch.from_numpy(np.load(val_c / "targets.npy")).float()

        entry: dict[str, Any] = {"depths": depths, "results": {}}
        print(f"  {'depth':>7}{'block':>7}{'val mAP':>10}{'sd':>8}   config")
        for i, depth in enumerate(depths):
            train_x = pooled_at_depth(train_c, i)
            val_x = pooled_at_depth(val_c, i)
            train_ds = DepthFeatures(train_x, train_y)
            val_ds = DepthFeatures(val_x, val_y)
            dim = train_x.shape[1]

            best_cfg, best_score, best_runs = None, -1.0, None
            for cfg in grid:
                runs = [fit_one(cfg, s, train_ds, val_ds, dim, args, device)
                        for s in range(args.seeds)]
                mean = float(np.mean([r[0] for r in runs]))
                if mean > best_score:
                    best_cfg, best_score, best_runs = cfg, mean, runs

            sd = float(np.std([r[0] for r in best_runs], ddof=1))
            entry["results"][str(depth)] = {
                "config": best_cfg, "val_map": best_score, "val_sd": sd,
                "epochs": [r[1] for r in best_runs],
                # The block index a relative depth resolves to, so that a reader
                # need not recompute it: depth d of an L-block encoder is block
                # round(d * L).
                "block": int(round(depth * 12)),
            }
            print(f"  {depth:>7.2f}{int(round(depth * 12)):>7}{best_score:>10.4f}"
                  f"{sd:>8.4f}   {best_cfg}")
        results["arms"][arm] = entry

    # The comparison, if a base and an adapted arm were both given.
    if len(args.arm) == 2:
        a, b = args.arm
        da, db = results["arms"][a], results["arms"][b]
        common = [d for d in da["depths"] if str(d) in db["results"]]
        print(f"\n{b} minus {a}, by depth")
        print(f"  {'depth':>7}{'change':>10}{'2 x sd':>10}")
        for d in common:
            ra, rb = da["results"][str(d)], db["results"][str(d)]
            change = rb["val_map"] - ra["val_map"]
            print(f"  {d:>7.2f}{change:>+10.4f}{2 * rb['val_sd']:>10.4f}")
        print("\n  Curves that coincide early and separate late mean continued")
        print("  pretraining altered the upper blocks and left the lower ones.")
        print("  Separation from the first depth measured means it altered the")
        print("  whole stack.")

    print(f"\n  A depth other than 1.0 outperforming it would mean the final")
    print(f"  block is not the right place to read for this task, which every")
    print(f"  result in this project and in the published literature assumes.")
    print(f"  Report the whole curve: selecting the best depth on validation is")
    print(f"  a selection, and Section 2.6 measures what selection costs here.")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "depth_probe.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwritten to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
